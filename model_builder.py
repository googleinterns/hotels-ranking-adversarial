from __future__ import print_function
import tensorflow as tf
import random
import six
import os
import numpy as np
import tensorflow_ranking as tfr
from tensorflow.python.framework import random_seed
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import training
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
from fgsm_calculations import *
from model_eval import *
import constants

class ModelBuilder:
    """Wrapping class to encapsulate functions to build model."""

    def __init__(self):

        # Parameters set by user during runtime.
        self.reference_number = 0
        self.answer_number = 0
        self.perturb_amount = 0

        # Flag to determine if using random noise.
        self.random_noise = False  

        # Flag to determine if this is first evaluation (in which case we save
        # tensor values).
        self.first_eval = True

        # Arrays with ranks of answers.
        self.ranking_array = []
        self.perturbed_ranking_array = []
        self.random_ranking_array = []

        # Array containing answer embeddings.
        self.embedded_features_tensor = None
        self.embedded_features_evaluated = [
            [0.0] * constants._FULL_EMBEDDING] * constants._LIST_SIZE

        # Array containing values for weights in first dense layer and gradient
        # of logits wrt to those weights. Used for FGSM calculation.
        self.grad_variable_pair_tensor = None
        self.grad_variable_pair_evaluated = [
            [[0] * constants._HIDDEN_LAYER_DIMS[0]] * constants._FULL_EMBEDDING] * 2

        # Embedding values post batch normalization. Used for FGSM calculation.
        self.normalized_features_evaluated = [
            [0.0] * constants._FULL_EMBEDDING] * constants._LIST_SIZE
        self.normalized_features = None

        # Textual features used for printing question/answers.
        self.query_features = None
        self.query_features_evaluated = None
        self.answer_features = None
        self.answer_features_evaluated = None

        # Labels used to determine number of answers/remove padding.
        self.labels_tensor = None
        self.labels_evaluated = [0] * constants._LIST_SIZE

        # Direction of random noise perturbation.
        self.random_noise_input = produce_random_noise()

        # Embedding for visualization.
        self.random_embedding = None
        self.fgsm_embedding = None

        self.optimizer = tf.compat.v1.train.AdagradOptimizer(
            learning_rate=constants._LEARNING_RATE)

    def context_feature_columns(self):
        """Returns context feature names to column definitions."""
        sparse_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key="query_tokens", vocabulary_file=constants._VOCAB_PATH)
        query_embedding_column = tf.feature_column.embedding_column(
            sparse_column, constants._EMBEDDING_DIMENSION)

        return {"query_tokens": query_embedding_column}

    def example_feature_columns(self):
        """Returns the example feature columns."""
        sparse_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key="document_tokens", vocabulary_file=constants._VOCAB_PATH)
        document_embedding_column = tf.feature_column.embedding_column(
            sparse_column, constants._EMBEDDING_DIMENSION)

        return {"document_tokens": document_embedding_column}

    def input_fn(self, path, num_epochs=None):
        """Input function used during traning."""
        # Ensures we don't add perturbation during training.
        self.perturb_on = tf.constant(
            False)  
        context_feature_spec = tf.feature_column.make_parse_example_spec(
            self.context_feature_columns().values())
        label_column = tf.feature_column.numeric_column(
            _LABEL_FEATURE, dtype=tf.int64, default_value=constants._PADDING_LABEL)
        example_feature_spec = tf.feature_column.make_parse_example_spec(
            list(self.example_feature_columns().values()) + [label_column])
        dataset = tfr.data.build_ranking_dataset(
            file_pattern=path,
            data_format=tfr.data.ELWC,
            batch_size=constants._BATCH_SIZE,
            list_size=constants._LIST_SIZE,
            context_feature_spec=context_feature_spec,
            example_feature_spec=example_feature_spec,
            reader=tf.data.TFRecordDataset,
            shuffle=False,
            num_epochs=num_epochs)
        features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        label = tf.squeeze(features.pop(constants._LABEL_FEATURE), axis=2)
        label = tf.cast(label, tf.float32)

        return features, label

    def make_transform_fn(self):
        def _transform_fn(features, mode):
            """Defines transform_fn."""
            context_features, example_features = tfr.feature.encode_listwise_features(
                features=features,
                context_feature_columns=self.context_feature_columns(),
                example_feature_columns=self.example_feature_columns(),
                mode=mode,
                scope="transform_layer")

            return context_features, example_features
        return _transform_fn

    def make_score_fn(self):
        """Returns a scoring function to build `EstimatorSpec`."""

        def _score_fn(context_features, group_features, mode, params, config):
            """Defines the network to score a group of documents."""
            with tf.compat.v1.name_scope("input_layer"):
                context_input = [
                    tf.compat.v1.layers.flatten(context_features[name])
                    for name in sorted(self.context_feature_columns())
                ]
                group_input = [
                    tf.compat.v1.layers.flatten(group_features[name])
                    for name in sorted(self.example_feature_columns())
                ]
                input_layer = tf.concat(context_input + group_input, 1)

            self.embedded_features_tensor = input_layer

            # We only want to save embedding on first predict run.
            if self.first_eval == False:  
                input_layer = tf.convert_to_tensor(
                    self.embedded_features_evaluated)

            # Perturb input if indicated.
            input_layer = tf.cond(
                self.perturb_on,
                lambda: self._perturbed_layer(),
                lambda: self._nonperturbed_layer(input_layer))  

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            cur_layer = input_layer
            cur_layer = tf.compat.v1.layers.batch_normalization(
                cur_layer,
                training=is_training,
                momentum=0.99)
            self.normalized_features = cur_layer 

            for i, layer_width in enumerate(
                    d for d in constants._HIDDEN_LAYER_DIMS):
                cur_layer = tf.compat.v1.layers.dense(
                    cur_layer, units=layer_width)
                cur_layer = tf.compat.v1.layers.batch_normalization(
                    cur_layer,
                    training=is_training,
                    momentum=0.99)
                cur_layer = tf.nn.relu(cur_layer)
                cur_layer = tf.compat.v1.layers.dropout(
                    inputs=cur_layer, rate=constants._DROPOUT_RATE, training=is_training)
            logits = tf.compat.v1.layers.dense(
                cur_layer, units=constants._GROUP_SIZE)
            
            self.logits_tensor = logits

            return logits

        return _score_fn

    def _nonperturbed_layer(self, layer):
        return layer

    def _perturbed_layer(self):
        """Adds noise to answer embeddings."""
        noise = get_perturbed_input(
            self, self.answer_number, self.perturb_amount)
        noise_input = np.add(self.embedded_features_evaluated, noise)
        if self.random_noise:
            self.random_embedding = noise_input
        else:
            self.fgsm_embedding = noise_input
        return tf.convert_to_tensor(noise_input, dtype=tf.float32)

    def eval_metric_fns(self):
        """Returns a dict from name to metric functions.

        This can be customized as follows. Care must be taken when handling padded
        lists.

        def _auc(labels, predictions, features):
          is_label_valid = tf_reshape(tf.greater_equal(labels, 0.), [-1, 1])
          clean_labels = tf.boolean_mask(tf.reshape(labels, [-1, 1], is_label_valid)
          clean_pred = tf.boolean_maks(tf.reshape(predictions, [-1, 1], is_label_valid)
          return tf.metrics.auc(clean_labels, tf.sigmoid(clean_pred), ...)
        metric_fns["auc"] = _auc

        Returns:
          A dict mapping from metric name to a metric function with above signature.
        """
        metric_fns = {}
        metric_fns.update({
            "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
                tfr.metrics.RankingMetricKey.NDCG, topn=topn)
            for topn in [1, 3, 5, 10]
        })

        return metric_fns

    def _train_op_fn(self, loss):
        """Defines train op used in ranking head."""

        update_ops = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.UPDATE_OPS)
        minimize_op = self.optimizer.minimize(
            loss=loss, global_step=tf.compat.v1.train.get_global_step())
        train_op = tf.group([update_ops, minimize_op])

        return train_op

    def train_and_eval_fn(self, model_fn):
        """Train and eval function used by `tf.estimator.train_and_evaluate`."""
        run_config = tf.estimator.RunConfig(
            save_checkpoints_steps=1000)
        ranker = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=constants._MODEL_DIR,
            config=run_config)

        def train_input_fn(): return self.input_fn(constants._TRAIN_DATA_PATH)
        def eval_input_fn(): return self.input_fn(
            constants._TEST_DATA_PATH, num_epochs=1)

        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=constants._NUM_TRAIN_STEPS)
        eval_spec = tf.estimator.EvalSpec(
            name="eval",
            input_fn=eval_input_fn,
            throttle_secs=15)

        return (ranker, train_spec, eval_spec)

    def run_training(self):
        """Runs training on the model."""
        # Define a loss function. To find a complete list of available loss 
        # functions or to learn how to add your own custom function please refer
        # to the tensorflow_ranking.losses module.
        _LOSS = tfr.losses.RankingLossKey.APPROX_NDCG_LOSS   
        loss_fn = tfr.losses.make_loss_fn(_LOSS)

        ranking_head = tfr.head.create_ranking_head(
            loss_fn=loss_fn,
            eval_metric_fns=self.eval_metric_fns(),
            train_op_fn=self._train_op_fn)

        model_fn = tfr.model.make_groupwise_ranking_fn(
            group_score_fn=self.make_score_fn(),
            transform_fn=self.make_transform_fn(),
            group_size=constants._GROUP_SIZE,
            ranking_head=ranking_head)

        ranker, train_spec, eval_spec = self.train_and_eval_fn(model_fn)
        tf.estimator.train_and_evaluate(ranker, train_spec, eval_spec)
        print("---------------------Training Complete---------------------")

        return ranker

    def predict_input_fn(self, path):
        """Input function used during predictions."""
        context_feature_spec = tf.feature_column.make_parse_example_spec(
            self.context_feature_columns().values())
        label_column = tf.feature_column.numeric_column(
            constants._LABEL_FEATURE,
            dtype=tf.int64,
            default_value=constants._PADDING_LABEL)
        example_feature_spec = tf.feature_column.make_parse_example_spec(
            list(self.example_feature_columns().values()) + [label_column])
        dataset = tfr.data.build_ranking_dataset(
            file_pattern=path,
            data_format=tfr.data.ELWC,
            batch_size=constants._BATCH_SIZE,
            list_size=constants._LIST_SIZE,
            context_feature_spec=context_feature_spec,
            example_feature_spec=example_feature_spec,
            reader=tf.data.TFRecordDataset,
            shuffle=True,
            num_epochs=1)
        features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

        label = tf.squeeze(features.pop(constants._LABEL_FEATURE), axis=2)
        label = tf.cast(label, tf.float32)
        self.pred_features = features
        self.query_features = features.get('query_tokens')
        self.answer_features = features.get('document_tokens')
        self.labels_tensor = label

        return features

    def custom_predict(self, perturb, ranker, input_fn,
                       predict_keys=None,
                       hooks=None,
                       checkpoint_path=None,
                       yield_single_examples=True):

        if not checkpoint_path:
            checkpoint_path = checkpoint_management.latest_checkpoint(
                ranker._model_dir)
        if not checkpoint_path:
            logging.info(
                'Could not find trained model in model_dir: {}, running '
                'initialization to predict.'.format(
                    ranker._model_dir))
        with tf.Graph().as_default() as g:

            self.perturb_on = tf.compat.v1.placeholder(tf.bool)

            random_seed.set_random_seed(ranker._config.tf_random_seed)
            ranker._create_and_assert_global_step(g)
            features, input_hooks = ranker._get_features_from_input_fn(
                input_fn, ModeKeys.PREDICT)
            estimator_spec = ranker._call_model_fn(
                features, None, ModeKeys.PREDICT, ranker.config)

            # Call to warm_start has to be after model_fn is called.
            ranker._maybe_warm_start(checkpoint_path)

            predictions = estimator_spec.predictions
            all_hooks = list(input_hooks)
            all_hooks.extend(list([]))

            self.grad_variable_pair_tensor = calculate_grad_var_pair(self)

            with training.MonitoredSession(
                    session_creator=training.ChiefSessionCreator(
                        checkpoint_filename_with_path=checkpoint_path,
                        master=ranker._config.master,
                        scaffold=estimator_spec.scaffold,
                        config=ranker._session_config),
                    hooks=all_hooks) as mon_sess:
                while not mon_sess.should_stop():
                    [preds_evaluated,
                     temp_query_features_evaluated,
                     temp_answer_features_evaluated,
                     temp_embedded_features_evaluated,
                     temp_labels_evaluated,
                     temp_normalized_features_evaluated, 
                     self.grad_variable_pair_evaluated,] = mon_sess.run([predictions,
                                                                         self.query_features,
                                                                         self.answer_features,
                                                                         self.embedded_features_tensor,
                                                                         self.labels_tensor,
                                                                         self.normalized_features, 
                                                                         self.grad_variable_pair_tensor,],
                                                                        {self.perturb_on: perturb})
                    # Save values for tensors during first nonperturbed evaluation to be 
                    # used in next execution.
                    if self.first_eval:  
                        self.query_features_evaluated = temp_query_features_evaluated
                        self.answer_features_evaluated = temp_answer_features_evaluated
                        self.embedded_features_evaluated = temp_embedded_features_evaluated
                        self.labels_evaluated = temp_labels_evaluated
                        self.normalized_features_evaluated = temp_normalized_features_evaluated
                        self.first_eval = False

                    if not yield_single_examples:
                        yield preds_evaluated
                    elif not isinstance(predictions, dict):
                        for pred in preds_evaluated:
                            yield pred
                    else:
                        for i in range(
                                self._extract_batch_length(preds_evaluated)):
                            yield {
                                key: value[i]
                                for key, value in six.iteritems(preds_evaluated)
                            }
