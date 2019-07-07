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

        predictions = []
        for question_chunks in self._dataset_reader.make_chunks(example, {'dataset_name':''}):
            question_instances = []
            for instance in self._dataset_reader.gen_question_instances(question_chunks):
                question_instances.append(instance)

            pred = self.predict_batch_instance(question_instances)[0]
            # Leaving only the original question ID for this dataset in order to run the original eval script.
            if pred['qid'].find('_q_') > -1:
                pred['qid'] = pred['qid'].split('_q_')[1]
            predictions.append(pred)

        formated_predictions = {pred['qid']:pred['best_span_str'] for pred in predictions}
        return formated_predictions
