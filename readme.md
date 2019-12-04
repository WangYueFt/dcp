# Deep Closest Point

## Prerequisites 
PyTorch>=1.0: https://pytorch.org

scipy>=1.2 

numpy

h5py

tqdm

TensorboardX: https://github.com/lanpa/tensorboardX

## Training

### DCP-v1

python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd

### DCP-v2

python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd

## Testing

### DCP-v1

python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --eval

or 

python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --eval --model_path=xx/yy

### DCP-v2

python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd --eval

or 

python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd --eval --model_path=xx/yy

where xx/yy is the pretrained model

## Citation
Please cite this paper if you want to use it in your work,

	@InProceedings{Wang_2019_ICCV,
	  title={Deep Closest Point: Learning Representations for Point Cloud Registration},
	  author={Wang, Yue and Solomon, Justin M.},
	  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
	  month = {October},
	  year={2019}
	}

## License
MIT License
