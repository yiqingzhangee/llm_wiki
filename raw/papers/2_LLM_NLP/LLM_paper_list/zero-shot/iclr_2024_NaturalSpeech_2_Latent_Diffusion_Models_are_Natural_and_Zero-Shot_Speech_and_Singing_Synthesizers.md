# _**NaturalSpeech 2**_ **: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers** 

**Kai Shen** _[∗]_ **, Zeqian Ju,** _[∗]_ **Xu Tan** _[∗]_ **, Yanqing Liu, Yichong Leng, Lei He Tao Qin, Sheng Zhao, Jiang Bian** Microsoft Research Asia & Microsoft Azure Speech `https://aka.ms/speechresearch` 

## **Abstract** 

Scaling text-to-speech (TTS) to large-scale, multi-speaker, and in-the-wild datasets is important to capture the diversity in human speech such as speaker identities, prosodies, and styles (e.g., singing). Current large TTS systems usually quantize speech into discrete tokens and use language models to generate these tokens one by one, which suffer from unstable prosody, word skipping/repeating issue, and poor voice quality. In this paper, we develop _NaturalSpeech 2_ , a TTS system that leverages a neural audio codec with residual vector quantizers to get the quantized latent vectors and uses a diffusion model to generate these latent vectors conditioned on text input. To enhance the zero-shot capability that is important to achieve diverse speech synthesis, we design a speech prompting mechanism to facilitate in-context learning in the diffusion model and the duration/pitch predictor. We scale NaturalSpeech 2 to large-scale datasets with 44K hours of speech and singing data and evaluate its voice quality on unseen speakers. NaturalSpeech 2 outperforms previous TTS systems by a large margin in terms of prosody/timbre similarity, robustness, and voice quality in a zero-shot setting, and performs novel zero-shot singing synthesis with only a speech prompt. Audio samples are available at `https://speechresearch.github.io/naturalspeech2` . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers_images/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers.pdf-0001-05.png)


**----- Start of picture text -----**<br>
Only in Training<br>Speech  𝒙<br>Training & Inference<br>IC In-Context Learning<br>Codec Decoder<br>Condition  𝒄 IC Diffusion Model Latent  𝒛<br>IC Duration/Pitch Predictor Codec Encoder<br>Phoneme Encoder   Speech  𝒙<br>Text  𝒚<br>**----- End of picture text -----**<br>


Figure 1: The overview of NaturalSpeech 2, with an audio codec encoder/decoder and a latent diffusion model conditioned on a prior (a phoneme encoder and a duration/pitch predictor). The details of in-context learning in the duration/pitch predictor and diffusion model are shown in Figure 3. 

> _∗_ The first three authors contributed equally to this work, and their names are listed in random order. Corresponding author: Xu Tan, `xuta@microsoft.com` 

Preprint. Work in progress. 

## **1 Introduction** 

Human speech is full of diversity, with different speaker identities (e.g., gender, accent, timbre), prosodies, styles (e.g., speaking, singing), etc. Text-to-speech (TTS) [1, 2] aims to synthesize natural and human-like speech with both good quality and diversity. With the development of neural networks and deep learning, TTS systems [3, 4, 5, 6, 7, 8, 9, 10, 11] have achieved good voice quality in terms of intelligibility and naturalness, and some systems (e.g., NaturalSpeech [11]) even achieves human-level voice quality on single-speaker recording-studio benchmarking datasets (e.g., LJSpeech [12]). Given the great achievements in speech intelligibility and naturalness made by the whole TTS community, now we enter a new era of TTS where speech diversity becomes more and more important in order to synthesize natural and human-like speech. 

Previous speaker-limited recording-studio datasets are not enough to capture the diverse speaker identities, prosodies, and styles in human speech due to limited data diversity. Instead, we can train TTS models on a large-scale corpus to learn these diversities, and as a by-product, these trained models can generalize to the unlimited unseen scenarios with few-shot or zero-shot technologies. Current large-scale TTS systems [13, 14, 15] usually quantize the continuous speech waveform into discrete tokens and model these tokens with autoregressive language models. This pipeline suffers from several limitations: 1) The speech (discrete token) sequence is usually very long (a 10s speech usually has thousands of discrete tokens) and the autoregressive models suffer from error propagation and thus unstable speech outputs. 2) There is a dilemma between the codec and language model: on the one hand, the codec with token quantization (VQ-VAE [16, 17] or VQ-GAN [18]) usually has a low bitrate token sequence, which, although eases the language model generation, incurs information loss on the high-frequency fine-grained acoustic details; on the other hand, some improving methods [19, 20] use multiple residual discrete tokens to represent a speech frame, which increases the length of the token sequence multiple times if flattened and incurs difficulty in language modeling. 

In this paper, we propose _NaturalSpeech 2_ , a TTS system with latent diffusion models to achieve expressive prosody, good robustness, and most importantly strong zero-shot ability for speech synthesis. As shown in Figure 1, we first train a neural audio codec that converts a speech waveform into a sequence of latent vectors with a codec encoder, and reconstructs the speech waveform from these latent vectors with a codec decoder. After training the audio codec, we use the codec encoder to extract the latent vectors from the speech in the training set and use them as the target of the latent diffusion model, which is conditioned on prior vectors obtained from a phoneme encoder, a duration predictor, and a pitch predictor. During inference, we first generate the latent vectors from the text/phoneme sequence using the latent diffusion model and then generate the speech waveform from these latent vectors using the codec decoder. 

Table 1: The comparison between NaturalSpeech 2 and previous large-scale TTS systems. 

|Methods|Previous Systems [13, 14, 15]|NaturalSpeech 2|
|---|---|---|
|Representations|Discrete Tokens|Continuous Vectors|
|Generative Models|Autoregressive Models|Non-Autoregressvie/Diffusion|
|In-Context Learning|Both Text and Speech are Needed|Only Speech is Needed|
|Stability/Robustness?|�|�|
|One Acoustic Model?|�|�|
|Beyond Speech (e.g., Singing)?|�|�|



We elaborate on some design choices in NaturalSpeech 2 (shown in Table 1) as follows. 

- _Continuous vectors instead of discrete tokens._ To ensure the speech reconstruction quality of the neural codec, previous works usually quantize speech with multiple residual quantizers. As a result, the obtained discrete token sequence is very long (e.g., if using 8 residual quantizers for each speech frame, the resulting flattened token sequence will be 8 times longer), and puts much pressure on the acoustic model (autoregressive language model). Therefore, we use continuous vectors instead of discrete tokens, which can reduce the sequence length and increase the amount of information for fine-grained speech reconstruction (see Section 3.1). 

2 

- _Diffusion models instead of autoregressive models._ We leverage diffusion models to learn the complex distributions of continuous vectors in a non-autoregressive manner and avoid error propagation in autoregressive models (see Section 3.2). 

- _Speech prompting mechanisms for in-context learning._ To encourage the diffusion models to follow the characteristics in the speech prompt and enhance the zero-shot capability, we design speech prompting mechanisms to facilitate in-context learning in the diffusion model and pitch/duration predictors (see Section 3.3). 

Benefiting from these designs, NaturalSpeech 2 is more stable and robust than previous autoregressive models, and only needs one acoustic model (the diffusion model) instead of two-stage token prediction as in [21, 13], and can extend the styles beyond speech (e.g., singing voice) due to the duration/pitch prediction and non-autoregressive generation. 

We scale NaturalSpeech 2 to 400M model parameters and 44K hours of speech data, and generate speech with diverse speaker identities, prosody, and styles (e.g., singing) in zero-shot scenarios (given only a few seconds of speech prompt). Experiment results show that NaturalSpeech 2 can generate natural speech in zero-shot scenarios and outperform the previous strong TTS systems. Specifically, 1) it achieves more similar prosody with both the speech prompt and ground-truth speech; 2) it achieves comparable or better naturalness (in terms of CMOS) than the ground-truth speech on LibriSpeech and VCTK test sets; 3) it can generate singing voices in a novel timbre either with a short singing prompt, or interestingly with only a speech prompt, which unlocks the truly zero-shot singing synthesis (without a singing prompt). Audio samples can be found in `https://speechresearch.github.io/naturalspeech2` . 

## **2 Background** 

We introduce some background of NaturalSpeech 2, including the journey of text-to-speech synthesis on pursuing natural voice with high quality and diversity, neural audio codec models, and generative models for audio synthesis. 

## **2.1 TTS for Natural Voice: Quality and Diversity** 

Text-to-speech systems [2, 3, 4, 5, 6, 8, 9, 22, 10, 11] aim to generate natural voice with both high quality and diversity. While previous neural TTS systems can synthesize high-quality voice on single-speaker recording-studio datasets (e.g., LJSpeech [12]) and even achieve human-level quality (e.g., NaturalSpeech [11]), they cannot generate diverse speech with different speaker identities, prosodies, and styles, which are critical to ensure the naturalness of the synthesized speech. Thus, some recent works [13, 14, 15] attempt to scale the TTS systems to large-scale, multi-speaker, and in-the-wild datasets to pursue diversity. 

These systems usually leverage a neural codec to convert speech waveform into discrete token sequence and an autoregressive language model to generate discrete tokens from text, which suffers from a dilemma as shown in Table 2: 1) If the audio codec quantizes each speech frame into a single token with vector-quantizer (VQ) [16, 17, 18], this could ease the token generation in the language model due to short sequence length, but will affect the waveform reconstruction quality due to large compression rate or low bitrate. 2) If the audio codec quantizes each speech frame into multiple tokens with residual vector-quantizer (RVQ) [19, 20], this will ensure high-fidelity waveform reconstruction, but will cause difficulty in autoregressive model generation (error propagation and robust issues) due to the increased length in the token sequence. Thus, previous works such as AudioLM [21] leverage two-stage language models to first generate some coarse-grained tokens in each frame and then generate the remaining fine-grained tokens, which are complicated and incur cascaded errors. To avoid the above dilemma, we leverage a neural codec with continuous vectors and a latent diffusion model with non-autoregressive generation. 

## **2.2 Neural Audio Codec** 

Neural audio codec [23, 24, 19, 20] refers to a kind of neural network model that converts audio waveform into compact representations with a codec encoder and reconstructs audio waveform from these representations with a codec decoder. Since audio codec is traditionally used for audio 

3 

Table 2: The dilemma in the pipeline of discrete audio codec and autoregressive language model. 

|The Dilemma in Previous Systems|Single Token (VQ)<br>Multiple Tokens (RVQ)|
|---|---|
|Waveform Reconstruction (Discrete Audio Codec)<br>Hard<br>Easy<br>Token Generation (Autoregressive Language Model)<br>Easy<br>Hard||



compression and transmission, the compression rate is a critical metric and thus discrete tokens with low bitrate are usually chosen as the compact representations. For example, SoundStream [19] and Encodec [20] leverage vector-quantized variational auto-encoders (VQ-VAE) with multiple residual vector-quantizers to compress speech into multiple tokens, and have been used as the intermediate representations for speech/audio generation [21, 25, 13, 14, 15]. 

Although good reconstruction quality and low bitrate can be achieved by residual vector quantizers, they are mainly designed for compression and transmission purposes and may not be suitable to serve as the intermediate representation for speech/audio generation. The discrete token sequence generated by residual quantizers is usually very long ( _R_ times longer if _R_ residual quantizers are used), which is difficult for the language models to predict. Inaccurate predictions of discrete tokens will cause word skipping, word repeating, or speech collapse issues when reconstructing speech waveforms from these tokens. In this paper, we design a neural audio codec to convert speech waveform into continuous vectors instead of discrete tokens, which can maintain enough fine-grained details for precise waveform reconstruction without increasing the length of the sequence. 

## **2.3 Generative Models for Speech Synthesis** 

Different generative models have been applied to speech or audio synthesis, and among these, autoregressive models and diffusion models are the two most prominent methods. Autoregressive models have long been used in speech synthesis for waveform generation [23] or acoustic feature generation [3]. Inspired by the success of autoregressive models in language generation [26, 27, 28], autoregressive models have been applied in speech and audio generation [21, 25, 13, 14, 15]. Meanwhile, diffusion models have also been widely used in speech synthesis for waveform generation [29, 30] and acoustic feature generation [31, 32]. 

Although both models are based on iterative computation (following the left-to-right process or the denoising process), autoregressive models are more sensitive to sequence length and error propagation, which cause unstable prosody and robustness issues (e.g., word skipping, repeating, and collapse). Considering text-to-speech has a strict monotonic alignment and strong source-target dependency, we leverage diffusion models enhanced with duration prediction and length expansion, which are free from robust issues. 

## **3 NaturalSpeech 2** 

In this section, we introduce NaturalSpeech 2, a TTS system for natural and zero-shot voice synthesis with high fidelity/expressiveness/robustness on diverse scenarios (various speaker identities, prosodies, and styles). As shown in Figure 1, NaturalSpeech 2 consists of a neural audio codec (an encoder and a decoder) and a diffusion model with a prior (a phoneme encoder and a duration/pitch predictor). Since speech waveform is complex and high-dimensional, following the paradigm of regeneration learning [33], we first convert speech waveform into latent vectors using the audio codec encoder and reconstruct speech waveform from the latent vectors using the audio codec decoder. Next, we use a diffusion model to predict the latent vectors conditioned on text/phoneme input. We introduce the detailed designs of neural audio codec in Section 3.1 and the latent diffusion model in Section 3.2, as well as the speech prompting mechanism for in-context learning in Section 3.3. 

## **3.1 Neural Audio Codec with Continuous Vectors** 

We use a neural audio codec to convert speech waveform into continuous vectors instead of discrete tokens, as analyzed in Section 2.1 and 2.2. Audio codec with continuous vectors enjoys several 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers_images/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Encoder Residual Vector Quantizer Decoder<br>Audio Codec<br>…<br>…      …      …       …      …… V Quantized<br>… Latent Vector z<br>…<br>R To/From Diffusion Model<br>: Only in Training                  : Training & Inference  R: # Residual Quantizers    V: # Codebook Tokens<br>**----- End of picture text -----**<br>


Figure 2: The neural audio codec consists of an encoder, a residual vector-quantizer (RVQ), and a decoder. The encoder extracts the frame-level speech representations from the audio waveform, the RVQ leverages multiple codebooks to quantize the frame-level representations, and the decoder takes the quantized vectors as input and reconstructs the audio waveform. The quantized vectors also serve as the training target of the latent diffusion model. 

benefits: 1) Continuous vectors have a lower compression rate and higher bitrate than discrete tokens[2] , which can ensure high-quality audio reconstruction. 2) Each audio frame only has one vector instead of multiple tokens as in discrete quantization, which will not increase the length of the hidden sequence. 

As shown in Figure 2, our neural audio codec consists of an audio encoder, a residual vector-quantizer (RVQ), and an audio decoder: 1) The audio encoder consists of several convolutional blocks with a total downsampling rate of 200 for 16KHz audio, i.e., each frame corresponds to a 12 _._ 5ms speech segment. 2) The residual vector-quantizer converts the output of the audio encoder into multiple residual vectors following [19]. The sum of these residual vectors is taken as the quantized vectors, which are used as the training target of the diffusion model. 3) The audio decoder mirrors the structure of the audio encoder, which generates the audio waveform from the quantized vectors. The working flow of the neural audio codec is as follows. 

Audio Encoder : _h_ = _f_ enc( _x_ ) _,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers_images/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers.pdf-0005-05.png)


Audio Decoder : _x_ = _f_ dec( _z_ ) _,_ 

where _f_ enc, _f_ rvq, and _f_ dec denote the audio encoder, residual vector quantizer, and audio decoder. _x_ is the speech waveform, _h_ is the hidden sequence obtained by the audio encoder with a frame length of _n_ , and _z_ is the quantized vector sequence with the same length as _h_ . _i_ is the index of the speech frame, _j_ is the index of the residual quantizer and _R_ is the total number of residual quantizers, and _e[i] j_ is the embedding vector of the codebook ID obtained by the _j_ -th residual quantizer on the _i_ -th hidden frame (i.e., _h[i]_ ). The training of the neural codec follows the loss function in [19]. 

Actually, to obtain continuous vectors, we do not need vector quantizers, but just an autoencoder or variational autoencoder. However, for regularization and efficiency purposes, we use residual vector quantizers with a very large number of quantizers ( _R_ in Figure 2) and codebook tokens ( _V_ in Figure 2) to approximate the continuous vectors. By doing this, we have two benefits: 1) When training latent diffusion models, we do not need to store continuous vectors which are memory-cost. Instead, we just store the codebook embeddings and the quantized token IDs, which are used to derive the continuous vectors using Equation 1. 2) When predicting the continuous vectors, we can add an additional regularization loss on discrete classification based on these quantized token IDs (see _L_ ce _−_ rvq in Section 3.2). 

## **3.2 Latent Diffusion Model with Non-Autoregressive Generation** 

We leverage a diffusion model to predict the quantized latent vector _z_ conditioned on the text sequence _y_ . We leverage a prior model that consists of a phoneme encoder, a duration predictor, and a pitch 

> 2Since our task is not speech compression but speech synthesis, we do not need a high compression rate or a low bitrate. 

5 

predictor to process the text input and provide a more informative hidden vector _c_ as the condition of the diffusion model. 

**Diffusion Formulation** We formulate the diffusion (forward) process and denoising (reverse) process as a stochastic differential equation (SDE) [34], respectively. The forward SDE transforms the latent vectors _z_ 0 obtained by the neural codec (i.e., _z_ ) into Gaussian noises: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers_images/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers.pdf-0006-02.png)


where _wt_ is the standard Brownian motion, _t ∈_ [0 _,_ 1], and _βt_ is a non-negative noise schedule function. Then the solution is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers_images/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers.pdf-0006-04.png)


By properties of Ito’s integral, the conditional distribution of _zt_ given _z_ 0 is Gaussian: _p_ ( _zt|z_ 0) _∼ t t N_ ( _ρ_ ( _z_ 0 _, t_ ) _,_ Σ _t_ ), where _ρ_ ( _z_ 0 _, t_ ) = _e[−]_[1] 2 �0 _[β][s][ds] z_ 0 and Σ _t_ = _I − e[−]_ �0 _[β][s][ds]_ . 

The reverse SDE transforms the Gaussian noise back to data _z_ 0 with the following process: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers_images/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers.pdf-0006-07.png)


where _w_ ˜ is the reverse-time Brownian motion. Moreover, we can consider an ordinary differential equation (ODE) [34] in the reverse process: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers_images/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers.pdf-0006-09.png)


We can train a neural network _sθ_ to estimate the score _∇_ log _pt_ ( _zt_ ) (the gradient of the log-density of noisy data), and then we can sample data _z_ 0 by starting from Gaussian noise _z_ 1 _∼N_ (0 _,_ 1) and numerically solving the SDE in Equation 4 or ODE in Equation 5. In our formulation, the neural network _sθ_ ( _zt, t, c_ ) is based on WaveNet [23], which takes the current noisy vector _zt_ , the time step _t_ , and the condition information _c_ as input, and predicts the data _z_ ˆ0 instead of the score, which we ˆ found results in better speech quality. Thus, _z_ 0 = _sθ_ ( _zt, t, c_ ). The loss function to train the diffusion model is as follows. 

ˆ _L_ diff = E _z_ 0 _,t_ [ _||z_ 0 _− z_ 0 _||_ 2[2][+] _[ ||]_[Σ] _[−] t_[1][(] _[ρ]_[(ˆ] _[z]_[0] _[, t]_[)] _[ −][z][t]_[)] _[ −∇]_[log] _[ p][t]_[(] _[z][t]_[)] _[||]_[2] 2[+] _[ λ][ce][−][rvq][L]_[ce] _[−]_[rvq][]] _[,]_ (6) where the first term is the data loss, the second term is the score loss, and the predicted score is calculated by Σ _[−] t_[1][(] _[ρ]_[(ˆ] _[z]_[0] _[, t]_[)] _[ −][z][t]_[)][, which is also used for reverse sampling based on Equation 4 or 5] in inference. The third term _L_ ce _−_ rvq is a novel cross-entropy (CE) loss based on residual vectorquantizer (RVQ). Specifically, for each residual quantizer _j ∈_ [1 _, R_ ], we first get the residual vector _z_ ˆ0 _−_[�] _[j] i_ =1 _[−]_[1] _[e][i]_[, where] _[ e][i]_[is the ground-truth quantized embedding in the] _[ i]_[-th residual quantizer (] _[e][i]_[is] also introduced in Equation 1). Then we calculate the L2 distance between the residual vector with each codebook embedding in quantizer _j_ and get a probability distribution with a softmax function, and then calculate the cross-entropy loss between the ID of the ground-truth quantized embedding _ej_ and this probability distribution. _L_ ce _−_ rvq is the mean of the cross-entropy loss in all _R_ residual quantizers, and _λ_ ce _−_ rvq is set to 0.1 during training. 

**Prior Model: Phoneme Encoder and Duration/Pitch Predictor** The phoneme encoder consists of several Transformer blocks [35, 6], where the standard feed-forward network is modified as a convolutional network to capture the local dependency in phoneme sequence. Both the duration and pitch predictors share the same model structure with several convolutional blocks but with different model parameters. The ground-truth duration and pitch information is used as the learning target to train the duration and pitch predictors, with an L1 duration loss _L_ dur and pitch loss _L_ pitch. During training, the ground-truth duration is used to expand the hidden sequence from the phoneme encoder to obtain the frame-level hidden sequence, and then the ground-truth pitch information is added to the frame-level hidden sequence to get the final condition information _c_ . During inference, the corresponding predicted duration and pitch are used. 

The total loss function for the diffusion model is as follows: 

_L_ = _L_ diff + _L_ dur + _L_ pitch _._ 

(7) 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers_images/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers.pdf-0007-00.png)


**----- Start of picture text -----**<br>
Duration/Pitch Predictor Diffusion Model 𝒛𝟎<br>Duration/Pitch<br>FiLM<br>K/V K/V K/V<br>Q-K-V Attention Q-K-V Attention Q-K-V Attention<br>Q Q Q<br>Convolution Speech Prompt Encoder … WaveNet<br>N x m x N’<br>𝒛 [𝒑]<br>Phoneme Hidden 𝒛𝒕, 𝒕, 𝒄<br>**----- End of picture text -----**<br>


Figure 3: The speech prompting mechanism in the duration/pitch predictor and the diffusion model for in-context learning. During training, we use a random segment _z[u]_[:] _[v]_ of the target speech _z_ as the speech prompt _z[p]_ and use the diffusion model to only predict _z[\][u]_[:] _[v]_ . During inference, we use a reference speech of a specific speaker as the speech prompt _z[p]_ . Note that the prompt is the speech latent obtained by the codec encoder instead of the speech waveform. 

## **3.3 Speech Prompting for In-Context Learning** 

To facilitate in-context learning for better zero-shot generation, we design a speech prompting mechanism to encourage the duration/pitch predictor and the diffusion model to follow the diverse information (e.g., speaker identities) in the speech prompt. For a speech latent sequence _z_ , we randomly cut off a segment _z[u]_[:] _[v]_ with frame index from _u_ to _v_ as the speech prompt, and concatenate the remaining speech segments _z_[1:] _[u]_ and _z[v]_[:] _[n]_ to form a new sequence _z[\][u]_[:] _[v]_ as the learning target of the diffusion model. As shown in Figure 3, we use a Transformer-based prompt encoder to process the speech prompt _z[u]_[:] _[v]_ ( _z[p]_ in the figure) to get a hidden sequence. To leverage this hidden sequence as the prompt, we have two different strategies for the duration/pitch predictor and the diffusion model: 1) For the duration and pitch predictors, we insert a Q-K-V attention layer in the convolution layer, where the query is the hidden sequence of the convolution layer, and the key and value is the hidden sequence from the prompt encoder. 2) For the diffusion model, instead of directly attending to the hidden sequence from the prompt encoder that exposes too many details to the diffusion model and may harm the generation, we design two attention blocks: in the first attention block, we use _m_ randomly initialized embeddings as the query sequence to attend to the prompt hidden sequence, and get a hidden sequence with a length of _m_ as the attention results [36, 37, 38]; in the second attention block, we leverage the hidden sequence in the WaveNet layer as the query and the _m_ -length attention results as the key and value. We use the attention results of the second attention block as the conditional information of a FiLM layer [39] to perform affine transform on the hidden sequence of the WaveNet in the diffusion model. Please refer to Appendix B for the details of WaveNet architecture used in the diffusion model. 

## **3.4 Connection to NaturalSpeech** 

NaturalSpeech 2 is an advanced edition of the NaturalSpeech Series [11, 40]. Compared to its previous version NaturalSpeech [11], NaturalSpeech 2 has the following connections and distinctions. First, _goal_ . While both NaturalSpeech 1 and 2 aim at synthesizing natural voices (with good speech quality and diversity), their focuses are different. NaturalSpeech focuses on speech quality by synthesizing voices that are on par with human recordings and only tackling single-speaker recording-studio datasets (e.g., LJSpeech). NaturalSpeech 2 focuses on speech diversity by exploring the zero-shot synthesis ability based on large-scale, multi-speaker, and in-the-wild datasets. Second, _architecture_ . NaturalSpeech 2 keeps the basic components in NaturalSpeech, such as the encoder and decoder for waveform reconstruction, and the prior module (phoneme encoder, duration/pitch predictor). However, it leverages 1) a diffusion model to increase the modeling power to capture the complicated and diverse data distribution in large-scale speech datasets, 2) a residual vector quantizer to regularize the latent vectors to trade off the reconstruction quality and prediction difficulty, and 3) a speech prompting mechanism to enable zero-shot ability that is not covered in single-speaker synthesis system. 

7 

## **4 Experimental Settings** 

In this section, we introduce the experimental settings to train and evaluate NaturalSpeech 2, including the dataset, model configuration, baselines for comparison, training and inference, and evaluation metrics. 

## **4.1 Datasets** 

**Training Dataset** To train the neural audio codec and the diffusion model, we use the English subset of Multilingual LibriSpeech (MLS) [41] as the training data, which contains 44K hours of transcribed speech data derived from LibriVox audiobooks. The number of distinct speakers is 2742 for males and 2748 for females respectively. The sample rate is 16KHz for all speech data. The input text sequence is first converted into a phoneme sequence using grapheme-to-phoneme conversion [42] and then aligned with speech using our internal alignment tool to obtain the phoneme-level duration. The frame-level pitch sequence is extracted from the speech using PyWorld[3] . 

**Evaluation Dataset** We employ two benchmark datasets for evaluation: 1) LibriSpeech [43] test-clean, which contains 40 distinct speakers and 5 _._ 4 hours of annotated speech data. 2) VCTK dataset [44], which contains 108 distinct speakers. For LibriSpeech test-clean, we randomly sample 15 utterances for each speaker and form a subset of 600 utterances for evaluation. For VCTK, we randomly sample 5 utterances for each speaker, resulting in a subset of 540 utterances for evaluation. Specifically, to synthesize each sample, we randomly select a different utterance of the same speaker and crop it into a _σ_ -second audio segment to form a _σ_ -second prompt. Note that both the speakers in LibriSpeech test-clean and VCTK are not seen during training. Thus, we aim to conduct zero-shot speech synthesis. 

The singing datasets follow a similar process in the speech dataset, and the details are shown in Section 5.6. 

## **4.2 Model Configuration and Comparison** 

**Model Configuration** The phoneme encoder is a 6-layer Transformer [35] with 8 attention heads, 512 embedding dimensions, 2048 1D convolution filter size, 9 convolution 1D kernel size, and 0.1 dropout rate. The pitch and duration predictor share the same architecture of 30-layer 1D convolution with ReLU activation and layer normalization, 10 Q-K-V attention layers for in-context learning, which have 512 hidden dimensions and 8 attention heads and are placed every 3 1D convolution layers. We set the dropout to 0.5 in both duration and pitch predictors. For the speech prompt encoder, we use a 6-layer Transformer with 512 hidden size, which has the same architecture as the phoneme encoder. As for the _m_ query tokens in the first Q-K-V attention in the prompting mechanism in the diffusion model (as shown in Figure 3), we set the token number _m_ to 32 and the hidden dimension to 512. 

The diffusion model contains 40 WaveNet layers [23], which consist of 1D dilated convolution layers with 3 kernel size, 1024 filter size, and 2 dilation size. Specifically, we use a FiLM layer [39] at every 3 WaveNet layers to fuse the condition information processed by the second Q-K-V attention in the prompting mechanism in the diffusion model. The hidden size in WaveNet is 512, and the dropout rate is 0.2. 

More details of the model configurations are shown in Appendix A. 

**Model Comparison** We choose the previous zero-shot TTS model YourTTS [45] as the baseline, with the official code and pre-trained checkpoint[4] , which is trained on VCTK [44], LibriTTS [46] and TTS-Portuguese [47]. We also choose VALL-E [13] that is based on discrete audio codec and autoregressive language model for comparison, which can help demonstrate the advantages of the designs in NaturalSpeech 2. We directly collect some audio samples from its demo page for comparison. 

> 3 `https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder` 

> 4 `https://github.com/Edresson/YourTTS` 

8 

## **4.3 Model Training and Inference** 

We first train the audio codec using 8 NVIDIA TESLA V100 16GB GPUs with a batch size of 200 audios per GPU for 440K steps. We follow the implementation and experimental setting of SoundStream [19] and adopt Adam optimizer with 2 _e−_ 4 learning rate. Then we use the trained codec to extract the quantized latent vectors for each audio to train the diffusion model in NaturalSpeech 2. 

The diffusion model in NaturalSpeech 2 is trained using 16 NVIDIA TESLA V100 32GB GPUs with a batch size of 6K frames of latent vectors per GPU for 300K steps (our model is still underfitting and longer training will result in better performance). We optimize the models with the AdamW optimizer with 5 _e −_ 4 learning rate, 32k warmup steps following the inverse square root learning schedule. 

During inference, for the diffusion model, we find it beneficial to use a temperature _τ_ and sample the terminal condition _zT_ from _N_ (0 _, τ[−]_[1] _I_ ) [32]. We set _τ_ to 1 _._ 2[2] . To balance the generation quality and latency, we adopt the Euler ODE solver and set the diffusion steps to 150. 

## **4.4 Evaluation Metrics** 

We use both objective and subjective metrics to evaluate the zero-shot synthesis ability of NaturalSpeech 2 and compare it with baselines. 

**Objective Metrics** We evaluate the TTS systems with the following objective metrics: 

- _Prosody Similarity with Prompt._ We evaluate the prosody similarity (in terms of pitch and duration) between the generated speech and the prompt speech, which measures how well the TTS model follows the prosody in speech prompt in zero-shot synthesis. We calculate the prosody similarity with the following steps: 1) we extract phoneme-level duration and pitch from the prompt and the synthesized speech; 2) we calculate the mean, standard variation, skewness, and kurtosis [7] of the pitch and duration in each speech sequence; 3) we calculate the difference of the mean, standard variation, skewness, and kurtosis between each paired prompt and synthesized speech and average the differences among the whole test set. 

- _Prosody Similarity with Ground Truth._ We evaluate the prosody similarity (in terms of pitch and duration) between the generated speech and the ground-truth speech, which measures how well the TTS model matches the prosody in the ground truth. Since there is correspondence between two speech sequences, we calculate the Pearson correlation and RMSE of the pitch/duration between the generated and ground-truth speech, and average them on the whole test set. 

- _Word Error Rate._ We employ an ASR model to transcribe the generated speech and calculate the word error rate (WER). The ASR model is a CTC-based HuBERT [48] pre-trained on Librilight [49] and fine-tuned on the 960 hours training set of LibriSpeech. We use the official code and checkpoint[5] . 

**Subjective Metrics** We conduct human evaluation and use the intelligibility score and mean opinion score as the subjective metrics: 

- _Intelligibility Score._ Neural TTS models often suffer from the robustness issues such as word skipping, repeating, and collapse issues, especially for autoregressive models. To demonstrate the robustness of NaturalSpeech 2, following the practice in [6], we use the 50 particularly hard sentences (see Appendix C) and conduct an intelligibility test. We measure the number of repeating words, skipping words, and error sentences as the intelligibility score. 

- _CMOS and SMOS._ Since synthesizing natural voices is one of the main goals of NaturalSpeech 2, we measure naturalness using comparative mean option score (CMOS) with 12 native speakers as the judges. We also use the similarity mean option score (SMOS) between the synthesized and prompt speech to measure the speaker similarity, with 6 native speakers as the judges. 

> 5 `https://huggingface.co/facebook/hubert-large-ls960-ft` 

9 

## **5 Results on Natural and Zero-Shot Synthesis** 

In this section, we conduct a series of experiments to compare the NaturalSpeech 2 with the baselines from the following aspects: 1) _Generation Quality_ , by evaluating the naturalness of the synthesized audio; 2) _Generation Similarity_ , by evaluating how well the TTS system follows prompts; 3) _Robustness_ , by calculating the WER and an additional intelligibility test. 

## **5.1 Generation Quality** 

We conduct CMOS test to evaluate the generation quality (i.e., naturalness). We randomly select 20 utterances from the LibriSpeech and VCTK tests and crop the prompt speech to 3s. To ensure high-quality generation, we use a speech scoring model [50] to filter the multiple samples generated by the diffusion model with different starting Gaussian noises _z_ 1. Table 3 shows a comparison of NaturalSpeech 2 against baseline YourTTS and the ground truth. We have several observations: 1) Nat- 

Table 3: The CMOS results (v.s. NaturalSpeech 2) on LibriSpeech and VCTK. 

||Setting<br>Ground Truth|LibriSpeech<br>+0_._04|VCTK<br>_−_0_._30|
|---|---|---|---|
||YourTTS|_−_0_._65|_−_0_._58|
||NaturalSpeech 2|**0.00**|**0.00**|



uralSpeech 2 is comparable to the ground-truth recording in LibriSpeech (+0 _._ 04 is regarded as on par) and achieves much better quality on VCTK datasets ( _−_ 0 _._ 30 is a large gap), which demonstrates the naturalness of the speech generated by NaturalSpeech 2 is high enough. 2) NaturalSpeech shows 0 _._ 65 and 0 _._ 58 CMOS gain over YourTTS in LibriSpeech and VCTK, respectively, which shows the superiority of NaturalSpeech 2 over this baseline. 

## **5.2 Generation Similarity** 

Table 4: The prosody similarity between synthesized and prompt speech in terms of the difference in mean (Mean), standard variation (Std), skewness (Skew), and kurtosis (Kurt) of pitch and duration. 

|**LibriSpeech**|Pitch<br>Duration|
|---|---|
||Mean_↓_<br>Std_↓_<br>Skew_↓_<br>Kurt_↓_<br>Mean_↓_<br>Std_↓_<br>Skew_↓_<br>Kurt_↓_|
|YourTTS<br>NaturalSpeech 2|10.52<br>7.62<br>0.59<br>1.18<br>0.84<br>**0.66**<br>0.75<br>3.70<br>**10.11**<br>**6.18**<br>**0.50**<br>**1.01**<br>**0.65**<br>0.70<br>**0.60**<br>**2.99**|
|**VCTK**|Pitch<br>Duration|
||Mean_↓_<br>Std_↓_<br>Skew_↓_<br>Kurt_↓_<br>Mean_↓_<br>Std_↓_<br>Skew_↓_<br>Kurt_↓_|
|YourTTS<br>NaturalSpeech 2|13.67<br>6.63<br>0.72<br>1.54<br>**0.72**<br>0.85<br>0.84<br>3.31<br>**13.29**<br>**6.41**<br>**0.68**<br>**1.27**<br>0.79<br>**0.76**<br>**0.76**<br>**2.65**|



We use two metrics to evaluate the speech similarity: 1) prosody similarity between the synthesized and prompt speech. 2) SMOS test. To evaluate the prosody similarity, we randomly sample one sentence for each speaker for both LibriSpeech test-clean and VCTK dataset to form the test sets. Specifically, to synthesize each sample, we randomly and independently sample the prompt speech with _σ_ = 3 seconds. Note that YourTTS has seen 97 speakers in VCTK in training, but we still compare NaturalSpeech 2 with YourTTS on all the speakers in VCTK (i.e., the 97 speakers are seen to YourTTS but unseen to NaturalSpeech 2). 

We apply the alignment tool to obtain phoneme-level duration and pitch and calculate the prosody similarity metrics between the synthesized speech and the prompt speech as described in Section 4.4. The results are shown in Table 4. We have the following observations: 1) NaturalSpeech 2 consistently outperforms the baseline YourTTS in both LibriSpeech and VCTK on all metrics, which demonstrates that our proposed NaturalSpeech 2 can mimic the prosody of prompt speech 

Table 5: The SMOS on LibriSpeech and VCTK respectively. 

||Setting<br>GroundTruth||LibriSpeech<br>3_._33|VCTK<br>3_._86|
|---|---|---|---|---|
||YourTTS||2_._03|2_._43|
||NaturalSpeech|2|**3.28**|**3.20**|



much better. 2) Although YourTTS has seen 97 from 108 speakers in VCTK dataset, our model can still outperform it by a large margin, which demonstrates the advantages of NaturalSpeech 2. 

10 

Furthermore, we also compare prosody similarity between synthesized and ground-truth speech in Appendix D. 

We further evaluate the speaker similarity using SMOS test. We randomly select 10 utterances from LibriSpeech and VCTK datasets respectively, following the setting in the CMOS test. The length of the prompt speech is set to 3s. The results are shown in Table 5. NaturalSpeech 2 outperforms YourTTS by 1 _._ 25 and 0 _._ 77 SMOS scores for LibriSpeech and VCTK, respectively, which shows that NaturalSpeech 2 is significantly better in speaker similarity. 

## **5.3 Robustness** 

We use the full test set of LibriSpeech and VCTK as described in Section 4.1 to synthesize the speech and compute the word error rate (WER) between the transcribed text and ground-truth text. To synthesize each sample, we use a 3-second prompt by randomly cropping the whole prompt speech. The results are shown in Table 6. We observe that: 1) NaturalSpeech 2 significantly outperforms YourTTS in LibriSpeech and VCTK, indicating better synthesis of high-quality and 

Table 6: Word error rate on LibriSpeech and VCTK. 

||Setting<br>Ground Truth|LibriSpeech<br>1.94|VCTK<br>9.49|
|---|---|---|---|
||YourTTS|7.10|14.80|
||NaturalSpeech 2|**2.26**|**6.99**|



robust speech. 2) Our synthesized speech is comparable to the ground-truth speech in LibriSpeech and surpasses that in VCTK. The higher WER results in VCTK may stem from a noisy environment and the lack of ASR model fine-tuning in that dataset. 

Table 7: The robustness of NaturalSpeech 2 and other autoregressive/non-autoregressive models on 50 particularly hard sentences. We conduct an intelligibility test on these sentences and measure the number of word repeating, word skipping, and error sentences. Each kind of word error is counted at once per sentence. 

|AR/NAR<br>Model|Repeats<br>Skips|Error Sentences<br>Error Rate|
|---|---|---|
||||
|AR<br>Tacotron [3]<br>Transformer TTS [5]|4<br>11<br>7<br>15|12<br>24%<br>17<br>34%|
||||
|NAR<br>FastSpeech [6]<br>0<br>0<br>0<br>0%<br>NaturalSpeech [11]<br>0<br>0<br>0<br>0%|||
||||
|NAR<br>NaturalSpeech 2|0<br>0|0<br>0%|



Autoregressive TTS models often suffer from alignment mismatch between phoneme and speech, resulting in severe word repeating and skipping. To further evaluate the robustness of the diffusionbased TTS model, we adopt the 50 particularly hard sentences in FastSpeech [6] to evaluate the robustness of the TTS systems. We can find that the non-autoregressive models such as FastSpeech [6], NaturalSpeech [11], and also NaturalSpeech 2 are robust for the 50 hard cases, without any intelligibility issues. As a comparison, the autoregressive models such as Tacotron [3], Transformer TTS [5], and VALL-E [13] will have a high error rate on these hard sentences. The comparison results are provided in Table 7. 

## **5.4 Comparison with Other TTS Systems** 

In this section, we compare NaturalSpeech 2 with the zeroshot TTS model VALL-E [13]. We directly download the first 16 utterances from VALL-E demo page[6] , which consists of 8 samples from LibriSpeech and 8 samples from VCTK. We evaluate the CMOS and SMOS in Table 8. 

Table 8: SMOS and CMOS results between NaturalSpeech 2 and VALL-E. 

||Setting||SMOS|CMOS|
|---|---|---|---|---|
||GroundTruth<br>VALL-E<br>NaturalSpeech|2|4.09<br>3.53<br>**3.83**|-<br>_−_0_._31<br>**0.00**|



From the results, we find that NaturalSpeech 2 outperforms VALL-E 3.53 VALL-E by 0.3 in SMOS and 0.31 in CMOS, respectively. NaturalSpeech 2 **3.83** The SMOS results show that NaturalSpeech 2 is significantly better in speaker similarity. The CMOS results demonstrate that the speech generated by NaturalSpeech 2 is much more natural and of higher quality. 

> 6 `https://valle-demo.github.io/` 

11 

## **5.5 Ablation Study** 

Table 9: The ablation study of NaturalSpeech 2. The prosody similarity between the synthesized and prompt speech in terms of the difference in the mean (Mean), standard variation (Std), skewness (Skew), and kurtosis (Kurt) of pitch and duration. “-" denotes the model can not converge. 

||Pitch<br>Duration|
|---|---|
||Mean_↓_<br>Std_↓_<br>Skew_↓_<br>Kurt_↓_<br>Mean_↓_<br>Std_↓_<br>Skew_↓_<br>Kurt_↓_|
|NaturalSpeech 2|**10.11**<br>**6.18**<br>**0.50**<br>**1.01**<br>**0.65**<br>**0.70**<br>**0.60**<br>**2.99**|
|w/o. diff prompt|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|
|w/o. dur/pitch prompt|21.69<br>19.38<br>0.63<br>1.29<br>0.77<br>0.72<br>0.70<br>3.70|
|w/o. CE loss|10.69<br>6.24<br>0.55<br>1.06<br>0.71<br>0.72<br>0.74<br>3.85|
|w/o. query attn|10.78<br>6.29<br>0.62<br>1.37<br>0.67<br>0.71<br>0.69<br>3.59|



In this section, we perform ablation experiments. 1) To study the effect of the speech prompt, we remove the Q-K-V attention layers in the diffusion (abbr. w/o. diff prompt), and the duration and pitch predictors (abbr. w/o. dur/pitch prompt), respectively. 2) To study the effect of the cross-entropy (CE) loss _L_ ce _−_ rvq based on RVQ, we disable the CE loss by setting _λce−rvq_ to 0 (abbr. w/o. CE loss). 3) To study the effectiveness of two Q-K-V attention in speech prompting for diffusion in Section 3.3, we remove the first attention that adopts _m_ randomly initialized query sequence to attend to the prompt hidden and directly use one Q-K-V attention to attend to the prompt hidden (abbr. w/o. query attn). We report the prosody similarity metric between synthesized and prompt speech in Table 9. More ablation results between synthesized and ground-truth speech are included in Appendix E. 

We have the following observations: 1) Disabling speech prompt in the diffusion model significantly degrades prosody similarity (e.g., from 10 _._ 11 to 21 _._ 69 for the mean of the pitch or even can not converge), highlighting its importance for high-quality TTS synthesis. 2) Disabling cross-entropy loss worsens performance, as the residual vector quantizer’s layer-wise cross entropy provides regularization for precise latent representations. 3) Disabling query attention strategy also degrades prosody similarity. In practice, we find that applying cross-attention to prompt hidden will leak details and thus mislead generation. 

In addition, since the prompt length is an important hyper-parameter for zero-shot TTS, we would like to investigate the effect of the prompt length. We follow the setting of _prosody similarity between synthesized and prompt speech_ in Section 5.2. Specifically, we vary the prompt length by _σ_ = _{_ 3 _,_ 5 _,_ 10 _}_ seconds and report the prosody similarity metrics of NaturalSpeech 2. The results are shown in Table 10. We observe that when the prompt is longer, the similarity between the generated speech and the prompt is higher for NaturalSpeech 2. It shows that the longer prompt reveals more details of the prosody, which help the TTS model to generate more similar speech. 

## **5.6 Zero-Shot Singing Synthesis** 

In this section, we explore NaturalSpeech 2 to synthesize singing voice in a zero-shot setting, either given a singing prompt or only a speech prompt. 

For singing data collection, we crawl a number of singing voices and their paired lyrics from the Web. For singing data preprocessing, we utilize a speech processing model to remove the backing vocal and accompaniment in the song, and an ASR model to filter out samples with misalignments. The dataset is then constructed using the same process as speech data, ultimately containing around 30 hours of singing data. The dataset is upsampled and mixed with speech data for singing experiments. 

We use speech and singing data together to train NaturalSpeech 2 with a 5 _e −_ 5 learning rate. In inference, we set the diffusion steps to 1000 for better performance. To synthesize a singing voice, we use the ground-truth pitch and duration from another singing voice, and use different singing prompts to generate singing voices with different singer timbres. Interestingly, we find that NaturalSpeech 2 can generate a novel singing voice using speech as the prompt. See the demo page[7] for zero-shot singing synthesis with either singing or speech as the prompt. 

> 7 `https://speechresearch.github.io/naturalspeech2` 

12 

Table 10: The NaturalSpeech 2 prosody similarity between the synthesized and prompt speech with different lengths in terms of the difference in the mean (Mean), standard variation (Std), skewness (Skew), and kurtosis (Kurt) of pitch and duration. 

|w),and kurtosis|(Kurt)ofpitch and duration.|
|---|---|
|**LibriSpeech**|Pitch<br>Duration|
||Mean_↓_<br>Std_↓_<br>Skew_↓_<br>Kurt_↓_<br>Mean_↓_<br>Std_↓_<br>Skew_↓_<br>Kurt_↓_|
|3s|10.11<br>6.18<br>0.50<br>1.01<br>0.65<br>0.70<br>0.60<br>2.99|
|5s|6.96<br>4.29<br>0.42<br>0.77<br>0.69<br>0.60<br>0.53<br>2.52|
|10s|6.90<br>4.03<br>0.48<br>1.36<br>0.62<br>0.45<br>0.56<br>2.48|
|**VCTK**|Pitch<br>Duration|
||Mean_↓_<br>Std_↓_<br>Skew_↓_<br>Kurt_↓_<br>Mean_↓_<br>Std_↓_<br>Skew_↓_<br>Kurt_↓_|
|3s|13.29<br>6.41<br>0.68<br>1.27<br>0.79<br>0.76<br>0.76<br>2.65|
|5s|14.46<br>5.47<br>0.63<br>1.23<br>0.62<br>0.67<br>0.74<br>3.40|
|10s|10.28<br>4.31<br>0.41<br>0.87<br>0.71<br>0.62<br>0.76<br>3.48|



## **5.7 Extension to Voice Conversion and Speech Enhancement** 

In this section, we extend NaturalSpeech 2 to another two speech synthesis tasks: 1) voice conversion and 2) speech enhancement. See the demo page[8] for zero-shot voice conversion and speech enhancement examples. 

## **5.7.1 Voice Conversion** 

Besides zero-shot text-to-speech and singing synthesis, NaturalSpeech 2 also supports zero-shot voice conversion, which aims to convert the source audio _zsource_ into the target audio _ztarget_ using the voice of the prompt audio _zprompt_ . Technically, we first convert the source audio _zsource_ into an informative Gaussian noise _z_ 1 using a _source-aware diffusion process_ and generate the target audio _ztarget_ using a _target-aware denoising process_ , shown as follows. 

**Source-Aware Diffusion Process.** In voice conversion, it is helpful to provide some necessary information from source audio for target audio in order to ease the generation process. Thus, instead of directly diffusing the source audio with some Gaussian noise, we diffuse the source audio into a starting point that still maintains some information in the source audio. Specifically, inspired by the stochastic encoding process in Diffusion Autoencoder [51], we obtain the starting point _z_ 1 from _zsource_ as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers_images/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers.pdf-0013-07.png)


where Σ _[−] t_[1][(] _[ρ]_[(ˆ] _[s][θ]_[(] _[z][t][, t, c]_[)] _[, t]_[)] _[ −][z][t]_[)][is][the][predicted][score][at] _[t]_[.][We][can][think][of][this][process][as][the] reverse of ODE (Equation 5) in the denoising process. 

**Target-Aware Denoising Process.** Different from the TTS which starts from the random Gaussian noise, the denoising process of voice conversion starts from the _z_ 1 obtained from the source-aware diffusion process. We run the standard denoising process as in the TTS setting to obtain the final target audio _ztarget_ , conditioned on _c_ and the prompt audio _zprompt_ , where _c_ is obtained from the phoneme and the duration sequence of the source audio and the predicted pitch sequence. 

As a consequence, we observe that NaturalSpeech 2 is capable of producing speech that exhibits similar prosody to the source speech, while also replicating the timbre specified by the prompt. 

> 8 `https://speechresearch.github.io/naturalspeech2` 

13 

## **5.7.2 Speech Enhancement** 

NaturalSpeech 2 can be extended to speech enhancement, which is similar to the extension of voice conversion. In this setting, we assume that we have the source audio _zsource[′]_[which contains] background noise ( _z[′]_ denotes the audio with background noise), the prompt with background noise _zprompt[′]_[for] _[ the source-aware diffusion process]_[, and the prompt without background noise] _[ z][prompt]_ for _target-aware denoising process_ . Note that _zsource[′]_[and] _[ z] prompt[′]_[have the same background noise.] 

To remove the background noise, firstly, we apply the _source-aware diffusion process_ by _zsource[′]_[and] _zprompt[′]_[and obtain the] _[ z]_[1][as in Equation 8.][The source audio’s duration and pitch are utilized in] this procedure. Secondly, we run the _target-aware denoising process_ to obtain the clean audio by _z_ 1 and the clean prompt _zprompt_ . Specifically, we use the phoneme sequence, duration sequence, and pitch sequence of the source audio in this procedure. As a result, we find that NaturalSpeech 2 can effectively eliminate background noise while simultaneously preserving crucial aspects such as prosody and timbre. 

## **6 Conclusion and Future Work** 

In this paper, we develop NaturalSpeech 2, a TTS system that leverages a neural audio codec with continuous latent vectors and a latent diffusion model with non-autoregressive generation to enable natural and zero-shot text-to-speech synthesis. To facilitate in-context learning for zero-shot synthesis, we design a speech prompting mechanism in the duration/pitch predictor and the diffusion model. By scaling NaturalSpeech 2 to 400M model parameters, 44K hours of speech, and 5K speakers, it can synthesize speech with high expressiveness, robustness, fidelity, and strong zero-shot ability, outperforming previous TTS systems. For future work, we will explore efficient strategies such as consistency models [52, 53] to speed up the diffusion model and explore large-scale speaking and singing voice training to enable more powerful mixed speaking/singing capability. 

**Broader Impacts** : Since NaturalSpeech 2 could synthesize speech that maintains speaker identity, it may carry potential risks in misuse of the model, such as spoofing voice identification or impersonating a specific speaker. We conducted the experiments under the assumption that the user agree to be the target speaker in speech synthesis. If the model is generalized to unseen speakers in the real world, it should include a protocol to ensure that the speaker approves the use of their voice and a synthesized speech detection model. 

14 

## **References** 

- [1] Paul Taylor. _Text-to-speech synthesis_ . Cambridge university press, 2009. 

- [2] Xu Tan, Tao Qin, Frank Soong, and Tie-Yan Liu. A survey on neural speech synthesis. _arXiv preprint arXiv:2106.15561_ , 2021. 

- [3] Yuxuan Wang, RJ Skerry-Ryan, Daisy Stanton, Yonghui Wu, Ron J Weiss, Navdeep Jaitly, Zongheng Yang, Ying Xiao, Zhifeng Chen, Samy Bengio, et al. Tacotron: Towards end-to-end speech synthesis. _Proc. Interspeech 2017_ , pages 4006–4010, 2017. 

- [4] Jonathan Shen, Ruoming Pang, Ron J Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, RJ Skerry-Ryan, et al. Natural TTS synthesis by conditioning WaveNet on mel spectrogram predictions. In _2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pages 4779–4783. IEEE, 2018. 

- [5] Naihan Li, Shujie Liu, Yanqing Liu, Sheng Zhao, and Ming Liu. Neural speech synthesis with Transformer network. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 33, pages 6706–6713, 2019. 

- [6] Yi Ren, Yangjun Ruan, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, and Tie-Yan Liu. FastSpeech: Fast, robust and controllable text to speech. In _NeurIPS_ , 2019. 

- [7] Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, and Tie-Yan Liu. FastSpeech 2: Fast and high-quality end-to-end text to speech. In _International Conference on Learning Representations_ , 2021. 

- [8] Yanqing Liu, Zhihang Xu, Gang Wang, Kuan Chen, Bohan Li, Xu Tan, Jinzhu Li, Lei He, and Sheng Zhao. DelightfulTTS: The Microsoft speech synthesis system for Blizzard challenge 2021. _arXiv preprint arXiv:2110.12612_ , 2021. 

- [9] Yanqing Liu, Ruiqing Xue, Lei He, Xu Tan, and Sheng Zhao. DelightfulTTS 2: Endto-end speech synthesis with adversarial vector-quantized auto-encoders. _arXiv preprint arXiv:2207.04646_ , 2022. 

- [10] Jaehyeon Kim, Jungil Kong, and Juhee Son. Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech. _arXiv preprint arXiv:2106.06103_ , 2021. 

- [11] Xu Tan, Jiawei Chen, Haohe Liu, Jian Cong, Chen Zhang, Yanqing Liu, Xi Wang, Yichong Leng, Yuanhao Yi, Lei He, et al. NaturalSpeech: End-to-end text to speech synthesis with human-level quality. _arXiv preprint arXiv:2205.04421_ , 2022. 

- [12] Keith Ito. The LJ speech dataset. `https://keithito.com/LJ-Speech-Dataset/` , 2017. 

- [13] Chengyi Wang, Sanyuan Chen, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, et al. Neural codec language models are zero-shot text to speech synthesizers. _arXiv preprint arXiv:2301.02111_ , 2023. 

- [14] Eugene Kharitonov, Damien Vincent, Zalán Borsos, Raphaël Marinier, Sertan Girgin, Olivier Pietquin, Matt Sharifi, Marco Tagliasacchi, and Neil Zeghidour. Speak, read and prompt: High-fidelity text-to-speech with minimal supervision. _arXiv preprint arXiv:2302.03540_ , 2023. 

- [15] Ruiqing Xue, Yanqing Liu, Lei He, Xu Tan, Linquan Liu, Edward Lin, and Sheng Zhao. Foundationtts: Text-to-speech for asr customization with generative language model. _arXiv preprint arXiv:2303.02939_ , 2023. 

- [16] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In _Proceedings of the 31st International Conference on Neural Information Processing Systems_ , pages 6309–6318, 2017. 

- [17] Ali Razavi, Aaron van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with VQ-VAE-2. In _Advances in neural information processing systems_ , pages 14866–14876, 2019. 

15 

- [18] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 12873–12883, 2021. 

- [19] Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. SoundStream: An end-to-end neural audio codec. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , 2021. 

- [20] Alexandre Défossez, Jade Copet, Gabriel Synnaeve, and Yossi Adi. High fidelity neural audio compression. _arXiv preprint arXiv:2210.13438_ , 2022. 

- [21] Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Olivier Teboul, David Grangier, Marco Tagliasacchi, and Neil Zeghidour. Audiolm: a language modeling approach to audio generation. _arXiv preprint arXiv:2209.03143_ , 2022. 

- [22] Jinglin Liu, Chengxi Li, Yi Ren, Feiyang Chen, and Zhou Zhao. DiffSinger: Singing voice synthesis via shallow diffusion mechanism. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 36, pages 11020–11028, 2022. 

- [23] Aäron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu. WaveNet: A generative model for raw audio. _arXiv preprint arXiv:1609.03499_ , 2016. 

- [24] Jean-Marc Valin and Jan Skoglund. LPCNet: Improving neural speech synthesis through linear prediction. In _ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pages 5891–5895. IEEE, 2019. 

- [25] Felix Kreuk, Gabriel Synnaeve, Adam Polyak, Uriel Singer, Alexandre Défossez, Jade Copet, Devi Parikh, Yaniv Taigman, and Yossi Adi. Audiogen: Textually guided audio generation. _arXiv preprint arXiv:2209.15352_ , 2022. 

- [26] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 

- [27] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_ , 1(8):9, 2019. 

- [28] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

- [29] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. DiffWave: A versatile diffusion model for audio synthesis. In _ICLR_ , 2021. 

- [30] Nanxin Chen, Yu Zhang, Heiga Zen, Ron J Weiss, Mohammad Norouzi, and William Chan. WaveGrad: Estimating gradients for waveform generation. In _ICLR_ , 2021. 

- [31] Myeonghun Jeong, Hyeongju Kim, Sung Jun Cheon, Byoung Jin Choi, and Nam Soo Kim. Diff-TTS: A denoising diffusion model for text-to-speech. _arXiv preprint arXiv:2104.01409_ , 2021. 

- [32] Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, and Mikhail Kudinov. GradTTS: A diffusion probabilistic model for text-to-speech. _arXiv preprint arXiv:2105.06337_ , 2021. 

- [33] Xu Tan, Tao Qin, Jiang Bian, Tie-Yan Liu, and Yoshua Bengio. Regeneration learning: A learning paradigm for data generation. _arXiv preprint arXiv:2301.08846_ , 2023. 

- [34] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In _International Conference on Learning Representations_ , 2020. 

- [35] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _Advances in Neural Information Processing Systems_ , pages 5998–6008, 2017. 

16 

- [36] Yequan Wang, Minlie Huang, Xiaoyan Zhu, and Li Zhao. Attention-based lstm for aspect-level sentiment classification. In _Proceedings of the 2016 conference on empirical methods in natural language processing_ , pages 606–615, 2016. 

- [37] Yuxuan Wang, Daisy Stanton, Yu Zhang, RJ Skerry-Ryan, Eric Battenberg, Joel Shor, Ying Xiao, Ye Jia, Fei Ren, and Rif A Saurous. Style tokens: Unsupervised style modeling, control and transfer in end-to-end speech synthesis. In _International Conference on Machine Learning_ , pages 5180–5189. PMLR, 2018. 

- [38] Dacheng Yin, Chuanxin Tang, Yanqing Liu, Xiaoqiang Wang, Zhiyuan Zhao, Yucheng Zhao, Zhiwei Xiong, Sheng Zhao, and Chong Luo. Retrievertts: Modeling decomposed factors for text-based speech insertion. _arXiv preprint arXiv:2206.13865_ , 2022. 

- [39] Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 32, 2018. 

- [40] Kai Shen, Zeqian Ju, Xu Tan, Yanqing Liu, Yichong Leng, Lei He, Tao Qin, Sheng Zhao, and Jiang Bian. Naturalspeech 2: Latent diffusion models are natural and zero-shot speech and singing synthesizers. _arXiv preprint arXiv:2304.09116_ , 2023. 

- [41] Vineel Pratap, Qiantong Xu, Anuroop Sriram, Gabriel Synnaeve, and Ronan Collobert. MLS: A large-scale multilingual dataset for speech research. _Proc. Interspeech 2020_ , pages 2757–2761, 2020. 

- [42] Hao Sun, Xu Tan, Jun-Wei Gan, Hongzhi Liu, Sheng Zhao, Tao Qin, and Tie-Yan Liu. Tokenlevel ensemble distillation for grapheme-to-phoneme conversion. In _INTERSPEECH_ , 2019. 

- [43] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. LibriSpeech: an ASR corpus based on public domain audio books. In _2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pages 5206–5210. IEEE, 2015. 

- [44] Christophe Veaux, Junichi Yamagishi, Kirsten MacDonald, et al. Superseded-CSTK VCTK corpus: English multi-speaker corpus for CSTK voice cloning toolkit. 2016. 

- [45] Edresson Casanova, Julian Weber, Christopher D Shulby, Arnaldo Candido Junior, Eren Gölge, and Moacir A Ponti. Yourtts: Towards zero-shot multi-speaker tts and zero-shot voice conversion for everyone. In _International Conference on Machine Learning_ , pages 2709–2720. PMLR, 2022. 

- [46] Heiga Zen, Viet Dang, Rob Clark, Yu Zhang, Ron J Weiss, Ye Jia, Zhifeng Chen, and Yonghui Wu. LibriTTS: A corpus derived from librispeech for text-to-speech. _Proc. Interspeech 2019_ , pages 1526–1530, 2019. 

- [47] Edresson Casanova, Arnaldo Candido Junior, Christopher Shulby, Frederico Santos de Oliveira, João Paulo Teixeira, Moacir Antonelli Ponti, and Sandra Aluísio. Tts-portuguese corpus: a corpus for speech synthesis in brazilian portuguese. _Language Resources and Evaluation_ , 56(3):1043–1055, 2022. 

- [48] Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , 29:3451–3460, 2021. 

- [49] Jacob Kahn, Morgane Riviere, Weiyi Zheng, Evgeny Kharitonov, Qiantong Xu, PierreEmmanuel Mazaré, Julien Karadayi, Vitaliy Liptchinsky, Ronan Collobert, Christian Fuegen, et al. Libri-light: A benchmark for asr with limited or no supervision. In _ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pages 7669–7673. IEEE, 2020. 

- [50] Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, et al. Wavlm: Large-scale self-supervised pretraining for full stack speech processing. _IEEE Journal of Selected Topics in Signal Processing_ , 16(6):1505–1518, 2022. 

17 

- [51] Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, and Supasorn Suwajanakorn. Diffusion autoencoders: Toward a meaningful and decodable representation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10619–10629, 2022. 

- [52] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. _arXiv preprint arXiv:2303.01469_ , 2023. 

- [53] Zhen Ye, Wei Xue, Xu Tan, Jie Chen, Qifeng Liu, and Yike Guo. Comospeech: One-step speech and singing voice synthesis via consistency model. _arXiv preprint arXiv:2305.06908_ , 2023. 

18 

## **A Model Details** 

|Module|Confguration|Value|#Parameters|
|---|---|---|---|
||Number of Residual VQ Blocks|16||
||Codebook size|1024||
|Audio Codec|Codebook Dimension|256|27M|
||Hop Size|200||
||Similarity Metric|L2||
||Transformer Layer|6||
||Attention Heads|8||
|Phoneme Encoder|Hidden Size<br>Conv1D Filter Size|512<br>2048|72M|
||Conv1D Kernel Size|9||
||Dropout|0.2||
||Conv1D Layers|30||
||Conv1D Kernel Size|3||
|Duration Predictor|Attention Layers<br>Attention Heads|10<br>8|34M|
||Hidden Size|512||
||Dropout|0.5||
||Conv1D Layers|30||
||Conv1D Kernel Size|5||
|Pitch Predictor|Attention Layers<br>Attention Heads|10<br>8|50M|
||Hidden Size|512||
||Dropout|0.5||
||Transformer Layer|6||
||Attention Heads|8||
|Speech Prompt Encoder|Hidden Size<br>Conv1D Filter Size|512<br>2048|69M|
||Conv1D Kernel Size|9||
||Dropout|0.2||
||WaveNet Layer|40||
||Attention Layers|13||
||Attention Heads|8||
|Diffusion Model|Hidden Size|512|183M|
||Query Tokens|32||
||Query Token Dimension|512||
||Dropout|0.2||
||Total||435M|



Table 11: The detailed model configurations of NaturalSpeech 2. 

## **B The Details of WaveNet Architecture in the Diffusion Model** 

As shown in Figure 4, the WaveNet consists of 40 blocks. Each block consists of 1) a dilated CNN with kernel size 3 and dilation 2, 2) a Q-K-V attention, and 3) a FiLM layer. In detail, we use Q-K-V attention to attend to the key/value obtained from the first Q-K-V attention module (from the speech prompt encoder) as shown in Figure 3. Then, we use the attention results to generate the scale and bias terms, which are used as the conditional information of the FiLM layer. Finally, we average the skip output results of each layer and calculate the final WaveNet output. 

19 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers_images/iclr_2024_NaturalSpeech_2_Latent_Diffusion_Models_are_Natural_and_Zero-Shot_Speech_and_Singing_Synthesizers.pdf-0020-00.png)


**----- Start of picture text -----**<br>
Layer k’s output<br>N Layers’ outputs<br>+<br>Layer k Layer k-1 …<br>tanh Gating σ<br>+<br>scale<br>FiLM scale * x + bias Linear ReLU<br>bias<br>x N<br>Conv1d 1x1<br>Condition c +<br>WaveNet output<br>Dilated Conv Q-K-V Attention<br>Q<br>K/V<br>Timestep t + from Prompt<br>Layer k’s input<br>**----- End of picture text -----**<br>


Figure 4: Overview of the WaveNet architecture in the diffusion model. 

## **C The 50 Particularly Hard Sentences** 

The 50 particularly hard sentences used in Section 5.3 are listed below: 

01. a 02. b 03. c 04. H 05. I 06. J 07. K 08. L 09. 22222222 hello 22222222 

10. S D S D Pass zero - zero Fail - zero to zero - zero - zero Cancelled - fifty nine to three - two - sixty four Total - fifty nine to three - two - 

11. S D S D Pass - zero - zero - zero - zero Fail - zero - zero - zero - zero Cancelled - four hundred and sixteen - seventy six - 

12. zero - one - one - two Cancelled - zero - zero - zero - zero Total - two hundred and eighty six - nineteen - seven - 

13. forty one to five three hundred and eleven Fail - one - one to zero two Cancelled - zero - zero to zero zero Total - 

14. zero zero one , MS03 - zero twenty five , MS03 - zero thirty two , MS03 - zero thirty nine , 

15. 1b204928 zero zero zero zero zero zero zero zero zero zero zero zero zero zero one seven ole32 

16. zero zero zero zero zero zero zero zero two seven nine eight F three forty zero zero zero zero zero six four two eight zero one eight 

20 

17. c five eight zero three three nine a zero bf eight FALSE zero zero zero bba3add2 - c229 - 4cdb - 

18. Calendaring agent failed with error code 0x80070005 while saving appointment . 

19. Exit process - break ld - Load module - output ud - Unload module - ignore ser - System error - ignore ibp - Initial breakpoint - 

20. Common DB connectors include the DB - nine , DB - fifteen , DB - nineteen , DB - twenty five , DB - thirty seven , and DB - fifty connectors . 

21. To deliver interfaces that are significantly better suited to create and process RFC eight twenty one , RFC eight twenty two , RFC nine seventy seven , and MIME content . 

22. int1 , int2 , int3 , int4 , int5 , int6 , int7 , int8 , int9 , 

23. seven _ ctl00 ctl04 ctl01 ctl00 ctl00 

24. Http0XX , Http1XX , Http2XX , Http3XX , 

25. config file must contain A , B , C , D , E , F , and G . 

26. mondo - debug mondo - ship motif - debug motif - ship sts - debug sts - ship Comparing local files to checkpoint files ... 

27. Rusbvts . dll Dsaccessbvts . dll Exchmembvt . dll Draino . dll Im trying to deploy a new topology , and I keep getting this error . 

28. You can call me directly at four two five seven zero three seven three four four or my cell four two five four four four seven four seven four or send me a meeting request with all the appropriate information . 

29. Failed zero point zero zero percent < one zero zero one zero zero zero zero Internal . Exchange . ContentFilter . BVT ContentFilter . BVT_log . xml Error ! Filename not specified . 

30. C colon backslash o one two f c p a r t y backslash d e v one two backslash oasys backslash legacy backslash web backslash HELP 

31. src backslash mapi backslash t n e f d e c dot c dot o l d backslash backslash m o z a r t f one backslash e x five 

32. copy backslash backslash j o h n f a n four backslash scratch backslash M i c r o s o f t dot S h a r e P o i n t dot 

33. Take a look at h t t p colon slash slash w w w dot granite dot a b dot c a slash access slash email dot 

34. backslash bin backslash premium backslash forms backslash r e g i o n a l o p t i o n s dot a s p x dot c s Raj , DJ , 

35. Anuraag backslash backslash r a d u r five backslash d e b u g dot one eight zero nine underscore P R two h dot s t s contains 

36. p l a t f o r m right bracket backslash left bracket f l a v o r right bracket backslash s e t u p dot e x e 

37. backslash x eight six backslash Ship backslash zero backslash A d d r e s s B o o k dot C o n t a c t s A d d r e s 

38. Mine is here backslash backslash g a b e h a l l hyphen m o t h r a backslash S v r underscore O f f i c e s v r 

39. h t t p colon slash slash teams slash sites slash T A G slash default dot aspx As always , any feedback , comments , 

40. two thousand and five h t t p colon slash slash news dot com dot com slash i slash n e slash f d slash two zero zero three slash f d 

41. backslash i n t e r n a l dot e x c h a n g e dot m a n a g e m e n t dot s y s t e m m a n a g e 

42. I think Rich’s post highlights that we could have been more strategic about how the sum total of XBOX three hundred and sixtys were distributed . 

43. 64X64 , 8K , one hundred and eighty four ASSEMBLY , DIGITAL VIDEO DISK DRIVE , INTERNAL , 8X , 

44. So we are back to Extended MAPI and C++ because . Extended MAPI does not have a dual interface VB or VB .Net can read . 

45. Thanks , Borge Trongmo Hi gurus , Could you help us E2K ASP guys with the following issue ? 

46. Thanks J RGR Are you using the LDDM driver for this system or the in the build XDDM driver ? 

47. Btw , you might remember me from our discussion about OWA automation and OWA readiness day a year ago . 

21 

48. empidtool . exe creates HKEY_CURRENT_USER Software Microsoft Office Common QMPersNum in the registry , queries AD , and the populate the registry with MS employment ID if available else an error code is logged . 

49. Thursday, via a joint press release and Microsoft AI Blog, we will announce Microsoft’s continued partnership with Shell leveraging cloud, AI, and collaboration technology to drive industry innovation and transformation. 

50. Actress Fan Bingbing attends the screening of ’Ash Is Purest White (Jiang Hu Er Nv)’ during the 71st annual Cannes Film Festival 

## **D Prosody Similarity with Ground Truth** 

To further investigate the quality of prosody, we follow the generation quality evaluation of _prosody similarity between synthesized and prompt speech_ in Section 5.2 and compare the generated speech with the ground-truth speech. We use Pearson correlation and RMSE to measure the prosody matching between generated and ground-truth speech. The results are shown in Table 12. We observe that NaturalSpeech 2 outperforms the baseline YourTTS by a large margin, which shows that our NaturalSpeech 2 is much better in prosody similarity. 

Table 12: The prosody similarity between the synthesized and ground-truth speech in terms of the correlation and RMSE on pitch and duration. 

|**LibriSpeech**|Pitch<br>Duration|
|---|---|
||Correlation_↑_<br>RMSE_↓_<br>Correlation_↑_<br>RMSE_↓_|
|YourTTS<br>NaturalSpeech 2|0.77<br>51.78<br>0.52<br>3.24<br>**0.81**<br>**47.72**<br>**0.65**<br>**2.72**|
|**VCTK**|Pitch<br>Duration|
||Correlation_↑_<br>RMSE_↓_<br>Correlation_↑_<br>RMSE_↓_|
|YourTTS<br>NaturalSpeech 2|0.82<br>42.63<br>0.55<br>2.55<br>**0.87**<br>**39.83**<br>**0.64**<br>**2.50**|



## **E Ablation Study** 

In this section, we also compare the prosody similarity between audio generated by the ablation model and the ground-truth speech in Table 13. Similar to the results of comparing the audio generated by the ablation model and prompt speech, we also have the following observations. 1) The speech prompt is most important to the generation quality. 2) The cross-entropy and the query attention strategy are also helpful in high-quality speech synthesis. 

Table 13: The ablation study of NaturalSpeech 2. The prosody similarity between the synthesized and ground-truth speech in terms of the correlation and RMSE on pitch and duration. “-" denotes that the model can not converge. 

||Pitch<br>Duration|
|---|---|
||Correlation_↑_<br>RMSE_↓_<br>Correlation_↑_<br>RMSE_↓_|
|NaturalSpeech 2|**0.81**<br>**47.72**<br>**0.65**<br>**2.72**|
|w/o. diff prompt|-<br>-<br>-<br>-|
|w/o. dur/pitch prompt|0.80<br>55.00<br>0.59<br>2.76|
|w/o. CE loss|0.79<br>50.69<br>0.63<br>2.73|
|w/o. query attn|0.79<br>50.65<br>0.63<br>2.73|



22 

