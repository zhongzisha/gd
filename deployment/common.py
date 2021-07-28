import paho.mqtt.client as mqtt
import paho.mqtt.publish as publisher
import pickle
import zlib


def send_msg(topic, msg_dict):
    msg = pickle.dumps(msg_dict, protocol=-1)
    msg = zlib.compress(msg)
    publisher.single(topic, msg, hostname="10.0.7.184", port=21883)


def load_msg(msg):
    msg_payload = zlib.decompress(msg.payload)
    msg_payload = pickle.loads(msg_payload)
    return msg_payload