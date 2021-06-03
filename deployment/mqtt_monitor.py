import pickle
import zlib
import paho.mqtt.client as mqtt

from common import load_msg

def on_connect(client, userdata, flags, rc):
    print('connected with result code ' + str(rc))
    client.subscribe("gd/detection")
    client.subscribe("gd/detection_results")


def on_message(client, userdata, msg):
    data_dict = load_msg(msg)
    print('topic', msg.topic)
    print(data_dict)


client = mqtt.Client("mqtt client for monitor")
client.on_connect = on_connect
client.on_message = on_message
client.connect("10.0.7.65", 21883, 60)

client.loop_forever()