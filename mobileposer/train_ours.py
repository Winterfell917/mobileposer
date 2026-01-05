from data_tic import *
from data_real import *
from trainner_ours import *
from model_tic import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = IMUData_Syn.load_data(folder_path=paths.amass_dir, step=2)
data_len = len(train_dataset['imu_rot'])
data_train = IMUData_Syn(rot=train_dataset['imu_rot'], acc=train_dataset['imu_acc'], seg_info=train_dataset['seg_info'],
                     head_acc=train_dataset['head_acc'], seq_len=128)

eval_dataset = IMUData_Real.load_data(folder_path=paths.real_dataset_processed_dir, step=1)
rot = eval_dataset['rot']
acc = eval_dataset['acc']
rot_gt = eval_dataset['rot_gt']
acc_gt = eval_dataset['acc_gt']
print(f"eval dataset: rot: {rot.shape}, acc: {acc.shape}, rot_gt: {rot_gt.shape}, acc_gt: {acc_gt.shape}")
data_eval = IMUData_Real(rot=rot, acc=acc, rot_gt=rot_gt, acc_gt=acc_gt, seg_info=eval_dataset['seg_info'], seq_len=128)

model = LSTMIC(n_input=config.imu_num*(3+3*3), n_output=config.imu_num*6).to(device)

optimizer = [torch.optim.Adam(model.parameters(), lr=1e-3)]

epoch=50
model_name = f'Ours_SynData_MODA'
ckpt_path = f'./data/checkpoint/calibrator/{model_name}'
log_dir = os.path.join("data/log", model_name)
os.makedirs(ckpt_path, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

trainner = TicTrainner(model=model, train_data=data_train, optimizer=optimizer, batch_size=128, val_data=data_eval, log_dir=log_dir)

print(model_name)
for i in range(epoch):
    trainner.run(epoch=1, data_shuffle=True)
    trainner.save(folder_path=ckpt_path, model_name=model_name)
    # trainner.log_export(f'./data/checkpoints/calibrator/log/{model_name}.xlsx')