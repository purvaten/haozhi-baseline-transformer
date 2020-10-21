# Region Proposal Interaction Network

Project page for neural physical prediction model

## Generate PHYRE data

You can use the following script to generate PHYRE data. 
```
python tools/gen_phyre.py --data data/dynamics/PHYRE_1fps_p20n80 --gen_data --subb 0 --sube 25
```

The default setting will generate 20 positive and 80 negative actions for each template (set the numbers here: https://github.com/HaozhiQi/RPIN-private/blob/master/tools/gen_phyre.py#L15-L16). The data generation part is slow due to the need of parsing objects (here: https://github.com/HaozhiQi/RPIN-private/blob/master/tools/gen_phyre.py#L78-L96). Therefore I would not recommend to run this script yourself if you just want to produce the results in the paper. Instead, you can download from this link: https://drive.google.com/file/d/1OhinT_CXg-WaTCJY_a4dBK2zRg9hCOYZ/view?usp=sharing (The file is not compressed, thus the final dataset size is 11G).

Its location should be ```data/dynamics/PHYRE_1fps_p20n80```.

## Train the Prediction Model

```
python train.py --cfg configs/phyre/pred/rpcin.yaml --gpus 0 --output ${OUTPUT_NAME}
```

The learning rate and mask loss weight is adjusted after submission. It should have better results.
