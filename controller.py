import os
from train_first_stage import main as train_first_stage
from train_second_stage import main as train_second_stage
from test_with_three_clips import main as test_with_three_clips

# /Users/sesena/Library/CloudStorage/GoogleDrive-rojenarda.2001@gmail.com/
LOCAL_DRIVE_PATH = os.path.join(os.path.expanduser('~'), 'Library', 'CloudStorage', 'GoogleDrive-rojenarda.2001@gmail.com')
COLAB_DRIVE_PATH = '/content/drive'

if os.path.exists(LOCAL_DRIVE_PATH):
    ECHO_FILE_ROOT = os.path.join(LOCAL_DRIVE_PATH, 'My Drive', 'echo')
else:
    ECHO_FILE_ROOT = os.path.join(COLAB_DRIVE_PATH, 'MyDrive', 'echo')

DATA_DIR = os.path.join(ECHO_FILE_ROOT, 'dataset', 'resized') # HFpEF DIR
UNIFORMER_WEIGHTS_DIR = os.path.join(ECHO_FILE_ROOT, 'models', 'uniformer_small_k400_16x8.pth')
COMET_API_KEY = ''
MODEL_PATH = os.path.join(ECHO_FILE_ROOT, 'models', 'coreecho')

stage_1 = True
stage_2 = False
test = False

if __name__ == '__main__':
    if not stage_1: quit()
    stage_1_args = [
        '--data_folder', DATA_DIR,
        '--pretrained_weights', UNIFORMER_WEIGHTS_DIR,
        '--project_name', 'coreecho-classifier-stage-1',
        '--model', 'uniformer_small',
        '--num_workers', '8',
        '--batch_size', '16',
        '--frames', '36',
        '--frequency', '4',
        '--learning_rate', '1e-4',
        '--weight_decay', '1e-4',
        '--lr_decay_rate', '0.1',
        '--val_n_clips_per_sample', '3',
        '--temp', '1.0',
        '--aug',
        '--epochs', '1',
        '--trial', '0',
        '--model_path', MODEL_PATH,
        '--comet_api_key', COMET_API_KEY,
        '--label_diff', 'binary'
    ]

    save_file_best = train_first_stage(stage_1_args)
    print(f'Stage 1 done: {save_file_best}')

    if not stage_2: quit()
    stage_2_args = [
        '--data_folder', DATA_DIR,
        '--project_name', 'coreeecho-training-stage-2',
        '--model', 'uniformer_small',
        '--num_workers', '8',
        '--batch_size', '16',
        '--frames', '36',
        '--frequency', '4',
        '--learning_rate', '1e-4',
        '--weight_decay', '1e-4',
        '--val_n_clips_per_sample', '3',
        '--aug',
        '--epochs', '1',
        '--trial', '0',
        '--model_path', MODEL_PATH,
        '--comet_api_key', COMET_API_KEY,
        '--pretrained_weights', save_file_best,
        '--label_diff', 'binary'
    ]

    save_file_best = train_second_stage(stage_2_args)
    print(f'Stage 2 done: {save_file_best}')

    if not test: quit()
    test_args = [
        '--data_folder', DATA_DIR,
        '--path_test_start_indexes', './test_start_indexes.pkl' ,
        '--path_save_test_files', os.path.join(ECHO_FILE_ROOT, 'coreecho', 'test_files'),
        '--model', 'uniformer_small',
        '--frames', '36',
        '--frequency', '4',
        '--num_workers', '4',
        '--project_name', 'coreecho-test',
    ]

    test_with_three_clips.main(test_args)
