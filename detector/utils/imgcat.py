# Shamefully stolen from https://github.com/fferri/py-imgcat

import os

from base64 import b64encode
from sys import stdout


def imgcat(data, width='auto', height='auto', preserveAspectRatio=False,
           inline=True, filename=''):
    """
    The width and height are given as a number followed by a unit, or the word "auto".
        N: N character cells.
        Npx: N pixels.
        N%: N percent of the session's width or height.
        auto: The image's inherent size will be used to determine an appropriate dimension.
    """
    buf = bytes()
    enc = 'utf-8'

    is_tmux = os.environ['TERM'].startswith('screen')

    # OSC
    buf += b'\033'
    if is_tmux:
        buf += b'Ptmux;\033\033'
    buf += b']'

    buf += b'1337;File='

    if filename:
        buf += b'name='
        buf += b64encode(filename.encode(enc))

    buf += b';size=%d' % len(data)
    buf += b';inline=%d' % int(inline)
    buf += b';width=%s' % width.encode(enc)
    buf += b';height=%s' % height.encode(enc)
    buf += b';preserveAspectRatio=%d' % int(preserveAspectRatio)
    buf += b':'
    buf += b64encode(data)

    # ST
    buf += b'\a'
    if is_tmux:
        buf += b'\033\\'

    buf += b'\n'

    stdout.buffer.write(buf)
    stdout.flush()

