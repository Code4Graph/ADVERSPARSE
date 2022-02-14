# ADVERSPARSE
This is the original pytorch implementation of ADVERSPARSE in the following paper: [ADVERSPARSE: AN ADVERSARIAL ATTACK FRAMEWORK FORDEEP SPATIAL-TEMPORAL GRAPH NEURAL NETWORKS, ICASSP 2020]


## Data prepration
Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g#list/path=%2F) links provided by DCRNN. You also need to download the file [sensor_graph](https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph) provided by DCRNN. 

## Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

### METR-LA
  
    python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5 --seq_length_x=12 ----seq_length_y=12

### PEMS-BAY
    
    python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5 --seq_length_x=12 ----seq_length_y=12

Note that the  readings  of  the  sensors  are  aggregated  into  5-minutes windows, therefore 15-minute-, 30-minute-, and 60-minutes-ahead predictions correspod to
the seq_length equal to 3, 6, 12 resepctively. You need to set the seq_length_x and seq_length_y based on your prediction when processing raw data.

## Pretrained step
    
    python train.py

Note that you may need to change some settings about the parameters in train.py based on the dataset you are running. For example, --data (data/PEMS-BAY or data/METR-LA), --seq_length(3, 6, or 12), ----num_nodes(PEMS-BAY:325 and METR-LA:207), --save(./garage/metr or ./garage/pems).

## ADMM training step

    python train_x.py

Note that you may need to change some settings about the parameters in train.py based on the dataset you are running. For example, --data (data/PEMS-BAY or data/METR-LA), --seq_length(3, 6, or 12), ----num_nodes(PEMS-BAY:325 and METR-LA:207), --save(./garage/metr or ./garage/pems), --best_pretrained_model(load the pretrained model whose file name has "best"), --keep(vary from 10% to 90%), --rho, --ADMMtimes, --epoch
