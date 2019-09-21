# pylint: disable=invalid-name
MULTIQA_VERSION = '0.1.0'

class MultiQA_DataSet():
    """

       """

    def __init__(self):
        pass

    def get_multiqa_version(self):
        return MULTIQA_VERSION

    def recursive_schema_compute(self,item, schema):
        if type(item) == list:
            for context in contexts:
                schema = self.recursive_schema_compute(context, schema)
        return schema

    def compute_schema(self, contexts):
        schema = {}
        for context in contexts:
            schema = self.recursive_schema_compute(context, schema)
        return schema

    def build_contexts(self):
        pass

    def build_header(self):
        pass

    def format_predictions(self, predictions):
        return predictions