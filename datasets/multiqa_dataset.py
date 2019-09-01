# pylint: disable=invalid-name
MULTIQA_VERSION = '0.1.0'

class MultiQA_DataSet():
    """
       A ``DatasetReader`` knows how to turn a file containing a dataset into a collection
       of ``Instance`` s.  To implement your own, just override the `_read(file_path)` method
       to return an ``Iterable`` of the instances. This could be a list containing the instances
       or a lazy generator that returns them one at a time.

       All parameters necessary to _read the data apart from the filepath should be passed
       to the constructor of the ``DatasetReader``.

       Parameters
       ----------
       lazy : ``bool``, optional (default=False)
           If this is true, ``instances()`` will return an object whose ``__iter__`` method
           reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.
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