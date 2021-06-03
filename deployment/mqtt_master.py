

from common import send_msg

msg_dict = {
    'job_id': 1,
    'tiffile': "/media/ubuntu/Working/rs/guangdong_aerial/aerial/220kvchangmianxiann31-n36.tif"
}

send_msg("gd/detection", msg_dict=msg_dict)