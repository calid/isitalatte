import tensorflow as tf
from isitalatte import graph_builder

class IsItALatteTest(tf.test.TestCase):

    def initAll(self):
        self.learning_rate   = 0.005
        self.batch_size      = 100
        self.training_epochs = 15

    def testIsItALatte(self):
        graph = graph_builder.build(
                self.learning_rate,
                self.batch_size,
                self.training_epochs)

        self.assertIsInstance(latte_graph, tf.Graph)
        for k in ('train_op', 'inputs', 'labels'):
            self.assertIn(k, graph.get_all_collection_keys())

if __name__ == '__main__':
    tf.test.main()
