from musicality import hdf5_getters


def test_read_beat():
    h5 = hdf5_getters.open_h5_file_read(
        "data/MillionSongSubset/A/A/A/TRAAAAW128F429D538.h5"
    )
    duration = hdf5_getters.get_duration(h5)
    beats_start = hdf5_getters.get_beats_start(h5)
    artist_name = hdf5_getters.get_artist_name(h5)
    song_name = hdf5_getters.get_title(h5)
    print(duration)
    print(beats_start)
    print(artist_name)
    print(song_name)
    h5.close()
