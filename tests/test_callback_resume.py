"""Real Transformers callback restoration; synthetic software state, not chemistry."""
import pytest


@pytest.mark.parametrize('restore, expected_counter, should_stop', [(False, 0, False), (True, 2, True)])
def test_nonzero_early_stop_patience_survives_checkpoint(tmp_path, restore, expected_counter, should_stop):
    torch=pytest.importorskip('torch')
    transformers=pytest.importorskip('transformers')
    from transformers import EarlyStoppingCallback, Trainer, TrainerControl, TrainerState, TrainingArguments

    old_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        saved_callback=EarlyStoppingCallback(early_stopping_patience=3)
        saved_callback.early_stopping_patience_counter=2
        saved_control=TrainerControl(should_training_stop=True, should_save=True)
        state=TrainerState(global_step=10,best_metric=1.0,best_model_checkpoint='software-only',
                           stateful_callbacks=[saved_callback,saved_control])
        checkpoint=tmp_path/'trainer_state.json'
        state.save_to_json(str(checkpoint))

        model=transformers.GPT2LMHeadModel(transformers.GPT2Config(
            vocab_size=16,n_positions=8,n_embd=8,n_layer=1,n_head=1))
        arguments=TrainingArguments(output_dir=str(tmp_path/'run'),use_cpu=True,report_to='none',
            eval_strategy='steps',eval_steps=1,save_strategy='steps',save_steps=1,
            metric_for_best_model='eval_loss',greater_is_better=False,load_best_model_at_end=True,
            restore_callback_states_from_checkpoint=restore,disable_tqdm=True)
        trainer=Trainer(model=model,args=arguments,eval_dataset=[{'input_ids':[1,2], 'labels':[1,2]}],
                        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
        trainer.state=TrainerState.load_from_json(str(checkpoint))
        trainer._load_callback_state()
        callback=next(c for c in trainer.callback_handler.callbacks if isinstance(c,EarlyStoppingCallback))
        assert callback.early_stopping_patience_counter==expected_counter
        # A saved SIGTERM stop flag must not immediately stop a resumed run.
        trainer.control=trainer.callback_handler.on_train_begin(arguments,trainer.state,trainer.control)
        assert not trainer.control.should_training_stop
        trainer.control=trainer.callback_handler.on_evaluate(arguments,trainer.state,trainer.control,
                                                             metrics={'eval_loss':2.0})
        assert trainer.control.should_training_stop is should_stop
    finally:
        torch.set_num_threads(old_threads)
