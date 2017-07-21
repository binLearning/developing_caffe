### Reshape()
```c++
template <typename Dtype>
void L2NormalizationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  squared_.Reshape(bottom[0]->num(), bottom[0]->channels(), 
      bottom[0]->height(), bottom[0]->width());
}
```
Setting up fc5 <br>
Top shape: 160 512 <br>
Creating Layer fc5_l2norm <br>
fc5_l2norm <- fc5 <br>
fc5_l2norm -> fc5_l2norm <br>
Setting up fc5_l2norm <br>
**Top shape: 160 512 1 1** <br>
Creating Layer fc5_l2norm_scale <br>
fc5_l2norm_scale <- fc5_l2norm <br>
fc5_l2norm_scale -> fc5_l2norm_scale <br>
Setting up fc5_l2norm_scale <br>
**Top shape: 160 512 1 1** <br>
Creating Layer fc6 <br>
fc6 <- fc5_l2norm_scale <br>
fc6 -> fc6 <br>
Setting up fc6 <br>
Top shape: 160 10557 <br>
Creating Layer softmax_loss <br>
softmax_loss <- fc6 <br>
softmax_loss <- label <br>
softmax_loss -> softmax_loss <br>
Setting up softmax_loss <br>
Top shape: (1) <br>
with loss weight 1 <br>
<br>
```c++
template <typename Dtype>
void L2NormalizationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->shape());
  squared_.Reshape(bottom[0]->shape());
}
```
Setting up fc5 <br>
Top shape: 160 512 <br>
Creating Layer fc5_l2norm <br>
fc5_l2norm <- fc5 <br>
fc5_l2norm -> fc5_l2norm <br>
Setting up fc5_l2norm <br>
**Top shape: 160 512** <br>
Creating Layer fc5_l2norm_scale <br>
fc5_l2norm_scale <- fc5_l2norm <br>
fc5_l2norm_scale -> fc5_l2norm_scale <br>
Setting up fc5_l2norm_scale <br>
**Top shape: 160 512** <br>
Creating Layer fc6 <br>
fc6 <- fc5_l2norm_scale <br>
fc6 -> fc6 <br>
Setting up fc6 <br>
Top shape: 160 10557 <br>
Creating Layer softmax_loss <br>
softmax_loss <- fc6 <br>
softmax_loss <- label <br>
softmax_loss -> softmax_loss <br>
Setting up softmax_loss <br>
Top shape: (1) <br>
with loss weight 1 <br>
