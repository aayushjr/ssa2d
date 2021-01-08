from keras.callbacks import Callback
from utils.sms import send
from utils.directory_utils import assert_existence


class LogHistory(Callback):
    """
    A Logger that log average performance per `display` steps.
    """

    def __init__(self, output_file='out/logs.txt', should_message=True, userids=('6235121378',),
                 message_carriers=('verizon',)):
        object.__init__(self)
        self.output_file = output_file
        self.should_message = should_message
        self.userids = userids
        self.message_carriers = message_carriers

    def on_epoch_end(self, epoch, logs=()):
        metrics = []
        for metric, value in logs.items():
            metrics.append("{}: {}".format(metric, round(value, 4)))

        with open(self.output_file, 'a+') as out_file:
            out_file.write(', '.join(metrics) + '\n')
        
        if self.should_message:
            out_string = "Epoch: {}\n".format(epoch)
            out_string += "\n".join(metrics)
            print(out_string)
            #send(out_string, userids=self.userids, carrier_names=self.message_carriers)
