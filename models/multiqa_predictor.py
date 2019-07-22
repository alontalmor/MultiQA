from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('multiqa_predictor')
class MultiQAPredictor(Predictor):
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        if 'header' in json_dict:
            return {}

        example = self._dataset_reader.combine_context(json_dict)
        self._dataset_reader._is_training = False

        predictions = []
        for question_chunks in self._dataset_reader.make_chunks(example, {'dataset_name':''}):
            question_instances = []
            for instance in self._dataset_reader.gen_question_instances(question_chunks):
                question_instances.append(instance)

            if len(question_instances) > 0:
                question_predictions = self.predict_batch_instance(question_instances)

                max_logit = -1000
                final_question_pred = {}
                for pred in question_predictions:
                    if pred['best_span_logit'] + pred['yesno_logit'] > max_logit:
                        max_logit = pred['best_span_logit'] + pred['yesno_logit']
                        final_question_pred['best_span_str'] = pred['best_span_str']

                # Leaving only the original question ID for this dataset in order to run the original eval script.
                final_question_pred['qid'] = pred['qid']
                if pred['qid'].find('_q_') > -1:
                    final_question_pred['qid'] = final_question_pred['qid'].split('_q_')[1]
                predictions.append(final_question_pred)
            else:
                qid = example['qas'][0]['qid']
                if qid.find('_q_') > -1:
                    qid = qid.split('_q_')[1]
                predictions.append({'qid':qid ,'best_span_str':''})

        formated_predictions = {pred['qid']:pred['best_span_str'] for pred in predictions}
        return formated_predictions
