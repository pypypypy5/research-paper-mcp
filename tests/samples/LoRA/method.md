We describe the simple design of LoRA and its practical benefits. The principles outlined here apply
to any dense layers in deep learning models, though we only focus on certain weights in Transformer
language models in our experiments as the motivating use case.
4.1 LOW-RANK-PARAMETRIZED UPDATE MATRICES
A neural network contains many dense layers which perform matrix multiplication. The weight
matrices in these layers typically have full-rank. When adapting to a specific task, Aghajanyan et al.
(2020) shows that the pre-trained language models have a low “instrisic dimension” and can still
learn efficiently despite a random projection to a smaller subspace. Inspired by this, we hypothe-
size the updates to the weights also have a low “intrinsic rank” during adaptation. For a pre-trained
weight matrix W0 ∈ Rd×k, we constrain its update by representing the latter with a low-rank de-
composition W0 + ∆W = W0 + BA, where B ∈ Rd×r , A ∈ Rr×k, and the rank r  min(d, k).
During training, W0 is frozen and does not receive gradient updates, while A and B contain trainable
parameters. Note both W0 and ∆W = BA are multiplied with the same input, and their respective
output vectors are summed coordinate-wise. For h = W0x, our modified forward pass yields:
h = W0x + ∆W x = W0x + BAx (3)
We illustrate our reparametrization in Figure 1. We use a random Gaussian initialization for A and
zero for B, so ∆W = BA is zero at the beginning of training. We then scale ∆W x by α
r , where α
is a constant in r. When optimizing with Adam, tuning α is roughly the same as tuning the learning
rate if we scale the initialization appropriately. As a result, we simply set α to the first r we try
and do not tune it. This scaling helps to reduce the need to retune hyperparameters when we vary
r (Yang & Hu, 2021).
A Generalization of Full Fine-tuning. A more general form of fine-tuning allows the training of
a subset of the pre-trained parameters. LoRA takes a step further and does not require the accumu-
lated gradient update to weight matrices to have full-rank during adaptation. This means that when
applying LoRA to all weight matrices and training all biases2, we roughly recover the expressive-
ness of full fine-tuning by setting the LoRA rank r to the rank of the pre-trained weight matrices. In
other words, as we increase the number of trainable parameters 3, training LoRA roughly converges
to training the original model, while adapter-based methods converges to an MLP and prefix-based
methods to a model that cannot take long input sequences.
No Additional Inference Latency. When deployed in production, we can explicitly compute and
store W = W0 + BA and perform inference as usual. Note that both W0 and BA are in Rd×k.
When we need to switch to another downstream task, we can recover W0 by subtracting BA and
then adding a different B′A′, a quick operation with very little memory overhead. Critically, this
2They represent a negligible number of parameters compared to weights.
3An inevitability when adapting to hard tasks.
4
guarantees that we do not introduce any additional latency during inference compared to a fine-tuned
model by construction.
4.2 APPLYING LORA TO TRANSFORMER
In principle, we can apply LoRA to any subset of weight matrices in a neural network to reduce the
number of trainable parameters. In the Transformer architecture, there are four weight matrices in
the self-attention module (Wq , Wk, Wv , Wo) and two in the MLP module. We treat Wq (or Wk, Wv )
as a single matrix of dimension dmodel × dmodel, even though the output dimension is usually sliced
into attention heads. We limit our study to only adapting the attention weights for downstream
tasks and freeze the MLP modules (so they are not trained in downstream tasks) both for simplicity
and parameter-efficiency.We further study the effect on adapting different types of attention weight
matrices in a Transformer in Section 7.1. We leave the empirical investigation of adapting the MLP
layers, LayerNorm layers, and biases to a future work.
Practical Benefits and Limitations. The most significant benefit comes from the reduction in
memory and storage usage. For a large Transformer trained with Adam, we reduce that VRAM
usage by up to 2/3 if r  dmodel as we do not need to store the optimizer states for the frozen
parameters. On GPT-3 175B, we reduce the VRAM consumption during training from 1.2TB to
350GB. With r = 4 and only the query and value projection matrices being adapted, the checkpoint
size is reduced by roughly 10,000× (from 350GB to 35MB)4. This allows us to train with signifi-
cantly fewer GPUs and avoid I/O bottlenecks. Another benefit is that we can switch between tasks
while deployed at a much lower cost by only swapping the LoRA weights as opposed to all the
parameters. This allows for the creation of many customized models that can be swapped in and out
on the fly on machines that store the pre-trained weights in VRAM. We also observe a 25% speedup
during training on GPT-3 175B compared to full fine-tuning5 as we do not need to calculate the
gradient for the vast majority of the parameters.
LoRA also has its limitations. For example, it is not straightforward to batch inputs to different tasks
with different A and B in a single forward pass, if one chooses to absorb A and B into W to eliminate
additional inference latency. Though it is possible to not merge the weights and dynamically choose
the LoRA modules to use for samples in a batch for scenarios where latency is not critical.