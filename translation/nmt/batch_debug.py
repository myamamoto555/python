#coding: utf-8


def _read_():

def _read_parallel_samples(src_filepath, trg_filepath, bos_id, eos_id):
  return zip(
      _read_single_samples(src_filepath, bos_id, eos_id),
      _read_single_samples(trg_filepath, bos_id, eos_id))


def generate_train_batch(
    src_filepath,
    trg_filepath,
    pad_id,
    bos_id,
    eos_id,
    batch_size,
    max_sample_length):

    samples = _read_parallel_samples(src_filepath, trg_filepath, bos_id, eos_id)
    samples = _filter_samples(samples, max_sample_length, max_length_ratio)
    samples = _arrange_samples(samples)
    samples = list(samples)
    batches = _split_samples(samples, batch_size)
    random.shuffle(batches)
    for batch in batches:

      yield _make_batch(batch, eos_id)
