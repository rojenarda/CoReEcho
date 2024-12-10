import os
import train_first_stage
import train_second_stage
import test_with_three_clips

ECHO_FILE_ROOT = os.path.join('drive', 'MyDrive', 'echo')
ECHONET_DYNAMIC_DATA_DIR = os.path.join(ECHO_FILE_ROOT, 'EchoNet-Dynamic')
UNIFORMER_WEIGHTS_DIR = os.path.join(ECHO_FILE_ROOT, 'models', 'uniformer_small_k400_16x8.pth')
COMET_API_KEY = ''
MODEL_PATH = os.path.join(ECHO_FILE_ROOT, 'models', 'coreecho')

if __name__ == '__main__':
    stage_1_args = [
        '--data_folder', ECHONET_DYNAMIC_DATA_DIR,
        '--pretrained_weights', UNIFORMER_WEIGHTS_DIR,
        '--project_name', 'coreecho-training-stage-1',
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
        '--epochs', '25',
        '--trial', '0',
        '--model_path', MODEL_PATH
    ]

    stage_2_args = [
        '--data_folder', ECHONET_DYNAMIC_DATA_DIR,
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
        '--epochs', '4',
        '--trial', '0',
        '--model_path', MODEL_PATH,
        '--comet_api_key', COMET_API_KEY
    ]

    test_args = [
        '--data_folder', ECHONET_DYNAMIC_DATA_DIR,
        '--path_test_start_indexes', './test_start_indexes.pkl' ,
        '--path_save_test_files', os.path.join(ECHO_FILE_ROOT, 'coreecho', 'test_files'),
        '--model', 'uniformer_small',
        '--frames', '36',
        '--frequency', '4',
        '--num_workers', '4',
        '--project_name', 'coreecho-test',
    ]

    if save_file_best := train_first_stage.main(stage_1_args):
        stage_2_args += [
            '--pretrained_weights', save_file_best,
        ]
        print('Stage 1 done')

        if save_file_best := train_second_stage.main(stage_2_args):
            test_args += [
                '--pretrained_weights', save_file_best,
            ]
            print('Stage 2 done')

            test_with_three_clips.main(test_args)
