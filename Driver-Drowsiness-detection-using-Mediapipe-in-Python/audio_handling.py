import av
import numpy as np
from pydub import AudioSegment


class AudioFrameHandler:
    """To play/pass custom audio based on some event"""

    def __init__(self, sound_file_path: str = ""):

        self.custom_audio = AudioSegment.from_file(file=sound_file_path, format="wav")
        self.custom_audio_len = len(self.custom_audio)

        self.ms_per_audio_segment: int = 20
        self.audio_segment_shape: tuple

        self.play_state_tracker: dict = {"curr_segment": -1}  # Currently playing segment
        self.audio_segments_created: bool = False
        self.audio_segments: list = []

    def prepare_audio(self, frame: av.AudioFrame):
        raw_samples = frame.to_ndarray()
        sound = AudioSegment(
            data=raw_samples.tobytes(),
            sample_width=frame.format.bytes,
            frame_rate=frame.sample_rate,
            channels=len(frame.layout.channels),
        )

        self.ms_per_audio_segment = len(sound)
        self.audio_segment_shape = raw_samples.shape

        self.custom_audio = self.custom_audio.set_channels(sound.channels)
        self.custom_audio = self.custom_audio.set_frame_rate(sound.frame_rate)
        self.custom_audio = self.custom_audio.set_sample_width(sound.sample_width)

        self.audio_segments = [
            self.custom_audio[i : i + self.ms_per_audio_segment]
            for i in range(0, self.custom_audio_len - self.custom_audio_len % self.ms_per_audio_segment, self.ms_per_audio_segment)
        ]
        self.total_segments = len(self.audio_segments) - 1  # -1 because we start from 0.

        self.audio_segments_created = True

    def process(self, frame: av.AudioFrame, play_sound: bool = False):

        """
        Takes in the current input audio frame and based on play_sound boolean value
        either starts sending the custom audio frame or dampens the frame wave to emulate silence.

        For eg. playing a notification based on some event.
        """

        if not self.audio_segments_created:
            self.prepare_audio(frame)

        raw_samples = frame.to_ndarray()
        _curr_segment = self.play_state_tracker["curr_segment"]

        if play_sound:
            if _curr_segment < self.total_segments:
                _curr_segment += 1
            else:
                _curr_segment = 0

            sound = self.audio_segments[_curr_segment]

        else:
            if -1 < _curr_segment < self.total_segments:
                _curr_segment += 1
                sound = self.audio_segments[_curr_segment]
            else:
                _curr_segment = -1
                sound = AudioSegment(
                    data=raw_samples.tobytes(),
                    sample_width=frame.format.bytes,
                    frame_rate=frame.sample_rate,
                    channels=len(frame.layout.channels),
                )
                sound = sound.apply_gain(-100)

        self.play_state_tracker["curr_segment"] = _curr_segment

        channel_sounds = sound.split_to_mono()
        channel_samples = [s.get_array_of_samples() for s in channel_sounds]

        new_samples = np.array(channel_samples).T

        new_samples = new_samples.reshape(self.audio_segment_shape)
        new_frame = av.AudioFrame.from_ndarray(new_samples, layout=frame.layout.name)
        new_frame.sample_rate = frame.sample_rate

        return new_frame
