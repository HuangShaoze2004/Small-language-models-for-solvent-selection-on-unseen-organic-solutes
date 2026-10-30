from solvent_select.llm import finish_training_segment


def test_resumed_runtime_counts_each_segment_once():
    record = {'status': 'interrupted', 'pid': 1, 'started': 100, 'global_step': 113}
    finish_training_segment(record, 80, 0)
    assert record['total_elapsed_seconds'] == 80
    resumed = {'status': 'completed', 'pid': 2, 'started': 200, 'global_step': 200,
               'segments': record['segments'].copy()}
    finish_training_segment(resumed, 60, 113)
    assert resumed['total_elapsed_seconds'] == 140
    assert [s['final_step'] - s['initial_step'] for s in resumed['segments']] == [113, 87]
