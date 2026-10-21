# **DiffPortrait3D: Controllable Diffusion for Zero-Shot Portrait View Synthesis** 

Yuming Gu[1,2] , You Xie[2] , Hongyi Xu[2] , Guoxian Song[2] , Yichun Shi[2] , Di Chang[1,2] ,Jing Yang[1] , Linjie Luo[2] 

1University of Southern California, 2ByteDance Inc. https://freedomgu.github.io/DiffPortrait3D 

_{_ yuminggu,dichang,jyang010 _}_ @usc.edu 

_{_ hongyixu,you.xie,guoxian.song,yichun.shi,linjie.luo _}_ @bytedance.com 

## **Abstract** 

_We present DiffPortrait3D, a conditional diffusion model that is capable of synthesizing 3D-consistent photo-realistic novel views from as few as a single in-the-wild portrait. Specifically, given a single RGB input, we aim to synthesize plausible but consistent facial details rendered from novel camera views with retained both identity and facial expression. In lieu of time-consuming optimization and finetuning, our zero-shot method generalizes well to arbitrary face portraits with unposed camera views, extreme facial expressions, and diverse artistic depictions. At its core, we leverage the generative prior of 2D diffusion models pre-trained on large-scale image datasets as our rendering backbone, while the denoising is guided with disentangled attentive control of appearance and camera pose. To achieve this, we first inject the appearance context from the reference image into the self-attention layers of the frozen UNets. The rendering view is then manipulated with a novel conditional control module that interprets the camera pose by watching a condition image of a crossed subject from the same view. Furthermore, we insert a trainable crossview attention module to enhance view consistency, which is further strengthened with a novel 3D-aware noise generation process during inference. We demonstrate state-ofthe-art results both qualitatively and quantitatively on our challenging in-the-wild and multi-view benchmarks._ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0001-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0001-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0001-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0001-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0001-12.png)


**----- Start of picture text -----**<br>
reference novel view synthesis<br>**----- End of picture text -----**<br>


Figure 1. Given a single portrait as reference (left), DiffPortrait3D is adept at producing high-fidelity and 3d-consistent novel view synthesis (right). Notably, without any finetuning, DiffPortrait3D is universally effective across a diverse range of facial portraits, encompassing, but not limited to, faces with exaggerated expressions, wide camera views, and artistic depictions. 

## **1. Introduction** 

Faithfully reconstructing the 3d appearance of human faces from a single 2D unconstrained portrait is a long-standing goal for computer vision, with a wide range of downstream applications in visual effects, digital avatars, 3D animation, and many others. In this work, we challenge ourselves to synthesize _high-fidelity consistent_ novel views from as few as a single portrait, with _high resemblance_ to the inputs 

in both individual appearance, expression and background content. Notably to the best of our knowledge, we are the first _zero-shot_ novel portrait synthesis work that supports versatile facial appearances and backgrounds, exaggerated expressions, wide views, and a plethora of artist styles. 

Long-range portrait view synthesis from sparse inputs requires a generative prior to hallucinating plausible scene features that are unobserved in the inputs. Recently, 3D aware generative adversarial network (GAN) [2, 5, 6, 12, 16, 36, 43, 55, 56] demonstrated striking quality and multi- 

1 

view-consistent image synthesis, by integrating 3D neural representations [35, 54] with style-based image generation [15, 26, 27]. Thereafter a line of work [3, 31, 39, 46, 57] has explored either optimization-based or encoder-based approaches to carefully invert the image into the latent or feature embedding of 3D GANs, and then synthesize novel views with 3D-aware generative priors. Nevertheless, almost all existing 3D-aware GANs are trained on limited image datasets. Hence when it comes to much more wild and nuanced portraits with large domain gap with the training distributions, GANs tend to struggle in faithfully depicting the 3D faces, resulting in loss of resemblance, corrupted geometry, or blurry extrapolation (see Figure 3, 4). 

With the recent advent of text-to-image diffusion models [20, 40, 44, 45], we have witnessed unprecedented diversity and stability in image synthesis exhibited by large diffusion models pre-trained on billions of images, such as Imagen [41] and Stable Diffusion (SD) [1]. We therefore aim to capitalize on the generative power of productionready diffusion models (SD in our work), for the task of portrait view synthesis. However, unlike previous 3D GANinversion works, simply inverting the reference image into a generative noise or a textual description does not naturally lift the image into a 3D scene, and it struggles to retain consistent appearances when deviating from the reference view. The introduction of ControlNet [59] enhances the controllability of Stable Diffusion by injecting localized spatial conditions. However, it remains unclear how to achieve appearance-disentangled view control such as in the paradigm of ControlNet. Moreover, without inherent 3D representation, the direct application of existing 2D image diffusion models to long-range animated view synthesis results in severe flickering artifacts. 

In this work, we propose _DiffPortrait3D_ , a novel zeroshot approach that lifts 2D diffusion model for synthesizing 3D consistent novel views from as few as a single portrait. Our key insight is to decompose the task into explicitly disentangled control of appearance and camera view. Specifically, we first utilize a trainable copy of the SD UNets to derive semantic appearance context from the reference image and then provide layer-by-layer contextual guidance to the self-attention modules of a locked SD network. This allows us to preserve the capability of the large diffusion models while generating images with retained reference characteristics regardless of the rendering views. On top of that, we further achieve view control by adding camera pose attention to the locked UNet decoder as done in ControlNet [59]. By design, the camera pose attention is intelligently extracted from an RGB portrait image of a proxy subject captured at the same view, to minimize appearance leakage from the condition image (e.g., shape and expression from landmarks). Additionally, to alleviate flickering artifacts when animating the views, we adopt a cross-view attention 

module as used in many video diffusion models [17, 21]. This ensures the unobservable region is completed in a consistent fashion. View consistency is further enhanced during inference with a novel 3D-aware noise generation process. 

With the locked parameters of Stable Diffusion, we finetuned our control modules in stages with multi-view synthetic dataset by PanoHead [2] and real-image Nersemble dataset [29]. Our method demonstrates native generalization capability to in-the-wild portraits without run-time fine-tuning. We extensively evaluate our framework on a few challenging benchmarks. DiffPortrait3D outperforms prior methods both quantitatively and qualitatively in terms of visual quality, resemblance, and view consistency. The contributions of our work can be summarized as: 

- A novel zero-shot view synthesis method that extends 2D Stable Diffusion for generating 3d consistent novel views given as little as a single portrait. 

- We demonstrate compelling fine-tuning-free novel view synthesis results given a single unconstrained portrait, regardless of its appearance, expression, pose, and style. 

- Explicitly disentangled control of appearance and camera view, enabling effective camera control with preserved identity and expression. 

- Long-range 3D view consistency with a cross-view attention module and 3D-aware noise generation. 

- Our code and model will be available for research purposes. 

## **2. Related Works** 

Our study focuses on the application of 2D diffusion models for zero-shot portrait novel view synthesis (NVS). Within this context, we undertake an extensive survey of progress in techniques related to novel view synthesis, categorized into regression-based and generative approaches. 

**Regression based NVS.** Facial NVS is attainable through the use of explicit parametric geometry priors, as demonstrated by 3D Morphable Models (3DMM) [13, 37, 48, 52, 61]. However, the limited parametric space of 3DMM poses challenges in faithfully depicting diverse facial expressions. Recent strides in Neural Radiance Fields (NeRF) [16, 22, 35, 58] have yielded high-fidelity results in novel view synthesis. Notably in the realm of portrait NVS, FDNeRF [58] constructed a NeRF model that integrates aligned features from inputs to generate novel view portraits. Nevertheless, achieving photo-realistic 3D-aware novel views with such models typically necessitates the availability of dense calibrated images. 

**Generative NVS with GAN** GANs [15] employ adversarial learning to synthesize images that faithfully capture the distribution of the training dataset. Previous studies have demonstrated the effectiveness of 2D GANs in portrait manipulation, employing techniques such as latent space exploration [8] and exemplar image utilization [25, 53]. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0003-00.png)


**----- Start of picture text -----**<br>
⨁ 𝜑()*(()<br>𝐼)*+ 𝓏, 𝐼./0 𝑄, 𝐾⨁, 𝑉⨁ 𝑄, 𝐾, 𝑉<br>+ ℰ 𝐴𝑡𝑡𝑛(() 𝐴𝑡𝑡𝑛(()<br>(b) Cross-Referenced Self-Attention<br>…<br>𝑧- … … ℝ ["×$, &][!][, '][!][, (][!] ℝ ["×'][!][×(][!][, $, &][!]<br>ResBlock Project In<br>TransBlock Multi-view<br>+ Attention<br>View-Consistency<br>Module Project Out<br>… … + … ℝ ["×$, &][!][, '][!][, (][!]<br>(c) View-Consistency Module<br>: SD U-Net module<br>+ 𝓕<br>View ControlNet 𝓕𝒄𝒂𝒎 Appearance Ref 𝓕𝒓𝒆𝒇 : view-consistency module<br>𝒟 : zero-conv layer<br>⨁ : concatenation<br>: attention concatenation<br>𝐼1 (a) Pipeline Overview : additive control<br>𝑅<br>**----- End of picture text -----**<br>


Figure 2. (a) Overview of our DiffPortrait3D framework. Given a single reference image _Iref ,_ we aim to synthesize its novel views as _IT_ at camera perspectives aligned with condition images _Icam._ We leverage a pre-trained LDM _F_ as our image synthesis backbone (middle), where its self-attention layers cross query the appearance context from _Iref_ via our appearance reference module _Fref_ (right). Our view control module (left) _Fcam_ derives additive view condition from _Icam_ and exerts on _F_ . Additionally, we plug in view consistency modules (dotted rectangles, middle) to _F_ to enhance multi-view coherence. During training, the images _Icam_ are rendered using an off-the-shelf 3D GAN renderer _R_ , where its camera perspectives are aligned with _IT_ . (b) The intermediate spatial features _φ_ ( _·_ ) sourced from _Iref_ are concatenated into the corresponding self-attention blocks in _F_ . (c) An attention mechanism is employed across the multi-view dimensions by our view-consistency module. 

Nevertheless, the absence of inherent 3D representations in these 2D GANs presents a challenge in maintaining 3D consistency for the task of NVS. 

Recent advancements on 3D aware GANs [2, 5, 6, 12, 16, 36, 43, 55, 56], built upon foundations of 2D GANs, have demonstrated striking quality and multi-viewconsistent image synthesis. These methodologies typically leverage StyleGAN2 [28] as a fundamental component, incorporating it with differential rendering and diverse 3D representations, such as signed distance functions as in StyleSDF [36] and tri-plane representations used by EG3D [6]. Thereafter a line of work [3, 31, 39, 46, 57] has explored either optimization-based or encoder-based approaches to carefully invert the image into the latent or feature embedding of 3D GANs, and then synthesize novel views with 3D-aware generative priors. It is noteworthy, however, that these methods heavily depend on a pre-trained 3D GAN generator and exhibit limitations in their capacity to generate unposed portraits with in-the-wild expressions, styles, and camera views. 

egy, harnessing powerful 2D diffusion models to improve the processes of 3D modeling and novel view synthesis. DreamFusion [38] pioneered this strategy by distilling a 2D text-to-image generation model for fine-tuning a NeRF model. GENVS [7] introduced a diffusion-based model explicitly tailored for 3D-aware generative novel view synthesis from a single input image. Their methodology involves modeling samples from the potential rendering distribution, effectively mitigating ambiguity and generating plausible novel views through the utilization of diffusion processes. Recent noteworthy study, Zero-1-to-3 [33, 34] utilizes a stable diffusion model to capture geometric priors derived from an extensive synthetic dataset, yielding highquality predictions. Moreover, Consistent123 [32], a caseaware approach, utilizes Zero-1-to-3 as 3D prior for the initial structural representation before generating high texture fidelity. However, it is crucial to note that these approaches primarily concentrate on general objects, resulting in a diminished quality when applied to portrait synthesis. 

## **3. Methods** 

**Diffusion Model based NVS** In lieu of directly confronting the intricacies of learning a 3D diffusion model, recent research endeavors have embraced an alternative strat- 

Given as few as a single RGB portrait image, denoted as _Iref_ , captured from any camera perspective, we aim to syn- 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0004-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0004-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0004-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0004-03.png)


**----- Start of picture text -----**<br>
reference Zero-1-to-3 GOAE Triplanenet EG3D-PTI ours<br>**----- End of picture text -----**<br>


Figure 3. **Qualitative comparison of novel view synthesis on in-the-wild images.** Compared to the baselines, our method shows superior generalization capability to novel view synthesis of wild portraits with unseen appearances, expressions and styles, even without any reliance on fine-tuning. 

thesize a new image _IT_ at a novel query view as indicated by a condition image _Icam_ . The synthesized image _IT_ should retain the expression and appearance of the foreground individual as well as the background context as in _Iref ,_ while follows the rendering view of _Icam._ Note that _Icam_ and _Iref_ could be of a completely different identity. 

Our proposed approach, DiffPortrait3D, leverages a latent diffusion model (LDM) as the backbone of our rendering framework, as depicted in Figure 2 (a) (Section 3.1). We then introduce an auxiliary appearance control branch (Section 3.2) to exert layer-by-layer guidance with local structures and textures from reference images _Iref ._ To enable effective camera control with _Icam,_ our view control module, designed in a fashion of ControlNet [59], implicitly derives camera pose from _Icam_ and inject to the diffusion process as an additive condition (Section 3.3). Lastly we discuss about enhancing view consistency with our integrated multi-view attentions, and noise generation with 3D awareness at inference (Section 3.4). 

## **3.1. Preliminaries** 

**Latent Diffusion Models.** Diffusion models [20, 44, 45] are generative models designed to synthesize desired data samples from Gaussian noise via removing noises iteratively. Latent diffusion models [40] are a class of diffusion models that operates in the encoded latent space of an autoencoder _D_ ( _E_ ( _·_ )) _,_ where _E_ and _D_ denotes the encoder and decoder respectively. Specifically, given an image _I_ and the text condition _ctext_ , the encoded image latent _z_ 0 = _E_ ( _I_ ) is diffused _T_ time steps into a Gaussian-distributed _zT ∼N_ (0 _,_ 1). The model is then trained to learn the reverse denoising process with the objective, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0004-09.png)


The _ϵθ_ is formulated as a trainable U-Net architecture with layers of intervened convolutions (ResBlock) and self/cross-attentions (TransBlock). In this paper, we build our network as a plug-and-play module to the recent state-ofthe-art text-to-image latent diffusion model, Stable Diffusion [1]. 

**ControlNet.** As introduced by [59], ControlNet effectively enhances latent diffusion models with spatially localized, task-specific image conditions. As its core, it replicates the original Stable Diffusion as a trainable side path, and adds additional “zero convolution” layers. The extra conditions outputted from the “zero convolution” layers are then added to the skipped connections of the SD-UNets. Let _cp_ be the extra condition, the noise prediction of U-Net with ControlNet then becomes _ϵθ_ � _zt, ctext, cp, t_ �. 

## **3.2. Appearance Reference Module** 

In order to synthesize a novel view of _Iref_ with LDM, one could try to condition the denoising with an “inverted” text condition _ctext_ [30]. However, providing a precise textual description of _Iref_ for LDM to comprehensively recover all its components is often a challenging undertaking. Alternatively, one could also condition _ϵθ_ on _Iref_ directly as a ControlNet. Such a design, however, tend to generate images predominantly influenced by the camera pose in _Iref ._ Inspired by [4, 51], we opt for integrating appearance attributes of the reference image _Iref_ into the UNet backbone as cross-referenced self-attentions. Note that to eliminate the harmful influence of inaccurate text description, we set _ctext_ empty and use the reference image _Iref_ as the only source of appearance. 

To illustrate our appearance reference module, let us denote the pretrained LDM as _F_ , where its self-attention is 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0005-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0005-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0005-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0005-03.png)


**----- Start of picture text -----**<br>
reference ground truth Zero-1-to-3 GOAE Triplanenet EG3D-PTI ours<br>**----- End of picture text -----**<br>


Figure 4. **Qualitative comparison of novel view synthesis on NeRSemble [29].** Our method achieves effective view control for novel synthesis with the best perceptual quality and retained identity and expression, even for portraits with exaggerated expressions and under substantial change of camera view for synthesis. 

calculated as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0005-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0005-07.png)


where _Q, K, V_ are the query, key, and value features projected from the spatial features _φ_ ( _zt_ ) with corresponding projection matrices respectively. 

To guide the denoising process with _Iref ,_ we adapt the self-attention mechanism within _F_ such that it is able to cross query the correlated local contents and textures from _E_ ( _Iref_ ), in addition to its own spatial features. Specifically we replicate _F_ into a trainable counterpart _Fref_ with _φref_ ( _·_ ) serving as intermediate representations within the UNet architecture. As depicted in Figure 2 (b), we then modify the vanilla self-attention in _F_ in a way that the spatial context _φref_ ( _E_ ( _Iref_ )) in the appearance branch _Fref_ is cross-queried layer by layer as, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0005-10.png)


where _⊕_ denotes concatenation. Note that we do not apply noise to _Iref ,_ ensuring meticulous transfer of referenced structure and appearance attributes into the novel portrait synthesis. We lock the parameters of SD-UNet _F_ , and train our appearance reference module _Fref_ with paired multiview images. 

Notable, when more reference images are available, e.g., in some multi-view capture settings, our appearance refer- 

ence module can be easily extended by concatenating multiple appearance contexts as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0005-14.png)


Our trained module is capable of seamlessly integrating the multi-view appearance clues into 3D-consistent appearance context (Figure 8). 

## **3.3. View Control Module** 

In this stage, we aim to attain control over the synthesis viewpoint without influencing either the derived appearance attributes by _Fref_ or the synthesis capability of a pretrained LDM _F_ . This naturally leads to the paradigm of ControlNet [59] where the additional view control is connected via “zero convolution” layers of a trainable LDM copy, with both _Fref_ and _F_ locked. Here we denote our view control module as _Fcam,_ to be trained with multi-view images. One straightforward design of _Fcam_ would be to employ the spatial feature maps extracted from the groundtruth target images as image conditions, such as landmarks, segmentation, or edges. We note that such “ground-truth” condition images are not available during inference and therefore the view is typically manipulated with images of a different identity. However, we argue that such condition images contain entangled semantic appearance information, such as shape and expression, which is likely to be passed along with the camera pose to _F_ . Herein, appearance leakage from the view condition image will be reflected on the novel view synthesis during inference. This artifact is more pronounced when _Iref_ and _Icam_ exhibit distinct appearance features. 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0006-00.png)


**----- Start of picture text -----**<br>
reference<br>novel view synthesis<br>w/o View-Consistency<br>w/o 3D aware noise<br>ours<br>**----- End of picture text -----**<br>


Figure 5. **Ablation on view consistency.** Excessive background variation and slight shading change across multiple novel views are observable without our view-consistency module. Our 3Daware noise, compared to random Gaussian noise, helps maintain structural coherence during view animation. 

Instead, we utilize a portrait image from a distinct random identity as the view condition, and generate novel-view images that mirror the head pose as in the condition portrait _Icam_ . Our design unifies the view manipulation setting in training and inference, and facilitates the natural disentanglement of view and appearance control. However, training ControlNet for cross-identity view control requires paired images at a identical view, and obtaining such data pairs is typically unfeasible in real-world capture settings. To address this hurdle, we leverage off-the-shelf 3D GAN renders _R_ ( _v, zv_ ), as exemplified in prior works [2, 6], to generate synthetic pose images _Icam_ . Here the _v_ denotes the camera parameters calibrated from the target image and _zv_ is a random Gaussian noise input to the 3D GAN. Since _Icam_ and _Iref_ possess substantial difference in expression and appearance, our view control module is therefore instructed to derive camera pose from _Icam_ only. Moreover, by design, the camera pose is directly interpreted by our view control module, allowing us to mimic the rendering view simply with an RGB image. This largely eases the cumbersomeness in feature processing of _Icam,_ e.g., landmarks detection or semantic parsing, which could be unreliable with heavy occlusion or under wide views. 

## **3.4. View Consistency Module** 

To this end, we have facilitated the generation of a novelview portrait via the seamless combination of an appearance reference module, a view ControlNet and a pretrained LDM. Nevertheless, achieving consistency in features across various views poses a significant challenge as many explanations exist for the unobservable region. Inspired by AnimateDiff [17], we introduce a view consis- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0006-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0006-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0006-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0006-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0006-09.png)


**----- Start of picture text -----**<br>
reference ours GOAE Triplanenet EG3D-PTI Zero-1-to-3<br>**----- End of picture text -----**<br>


Figure 6. **Reconstruction** . DiffPortrait3D demonstrates meticulous reconstruction of referenced appearance, even with side views and 3D cartoon styles, substantially outperforming the baseline methods. 

tency module that incorporates cross-view attention within a batch of views. Such a module employ an attention mechanism along the dimension of views to establish feature correlation among the multiple novel view synthesis. Similar to AnimateDiff, we integrate these view consistency modules into the up- and down-sampling blocks of the LDM _F_ , as depicted in Figure 2 (c). However, we note that such frame-wise modules were originally proposed for temporal coherence and as motion prior, trained with sequential video frames. In contrast, the animated view motion is purely defined by the sequence of _Icam._ Therefore, we trained our view-consistency modules with batches of randomly shuffled views, permitting the modules to focus on cross-view attentions in lieu of motion distribution. 

As illustrated in Figure 2 (c), we train our viewconsistency modules in groups of multi-view condition images _{Icam}_ with a shape of ( _B, V, C, H, W_ ), where _B_ and _V_ are the batch size and the number of views, while _C, H, W_ denote the number of image channels, image height and width respectively. We note that the appearance within each batch of generation is referenced from the same image _Iref_ . Inside _F_ , we reshape the input to ResBlocks and TransBlocks as ( _B ×V, C[′] , H[′] , W[′]_ ), where _C[′] , H[′] , W[′]_ represent the latent feature channel, height and width respectively. Following the operations of self- and crossattention, we then transform the layer input into a shape of ( _B × H[′] × W[′] , V, C[′]_ ), performing view-wise attention within the view consistency modules. 

**3D-aware inference.** It has been empirically observed that the image layout is formed in the early denoising steps. Therefore instead of denoising from multiple random Gaussian noises, structural and textural consistency is likely to be enhanced when synthesizing multiple novel views by initiating the denoising process from “3D-consistent” noise samples. We propose an efficient two-stage process to generate noise samples with 3D awareness. On our multi-view image dataset, we first trained a 3D-convolution based NVS model with inclusion of 3D feature field and neural feature rendering (please refer to the supplementary paper for details). We employ this NVS model to provide a proxy syn- 

6 

|Ours<br>Eg3D-PTI<br>GOAE<br>Triplanenet<br>POSE_↓_<br>-/0.0068/-<br>-/**0.0021**/-<br>-/0.0022/-<br>-/0.0134/-<br>LPIPS_↓_**0.02**/**0.23**/**0.02** 0.21/0.26/0.36<br>0.12/0.28/0.21<br>0.10/0.39/0.17<br>SSIM_↑_**0.92**/**0.62**/**0.93** 0.66/0.59/0.46<br>0.72/0.57/0.54<br>0.76/0.50/0.63<br>DIST_↓_**0.06**/**0.21**/**0.04** 0.21/0.24/0.27<br>0.15/0.25/0.20<br>0.15/0.29/0.19<br>ID_↑_<br>**0.95**/**0.70**/**0.92** 0.15/0.28/0.12<br>0.55/0.39/0.54<br>0.70/0.45/0.70<br>FID_↓_<br>**7.65**/**27.4**/**11.7** 33.13/56.2/95.0 54.10/84.8/92.0 63.54/112.1/88.0<br>(a)|w/o_Fref_<br>_Iref_<br>_Iref_<br>fnetuning unaligned aligned|
|---|---|
||LPIPS_↓_<br>0.55<br>0.28<br>**0.27**|
||SSIM_↑_<br>0.47<br>**0.68**<br>**0.68**|
||DIST_↓_<br>0.43<br>0.19<br>**0.18**|
||ID_↑_<br>0.21<br>**0.70**<br>**0.70**|
||FID_↓_<br>99.09<br>25.77<br>**25.37**|
||(b)|



Table 1. (a) Quantitative comparison of our method and GAN-based baselines, showing numerical results of reconstruction/novel view synthesis of NeRSemble [29], and reconstruction of in-the-wild test images( from left to right). For a fair comparison to our baselines, the evaluation is performed at the resolution of 256 _×_ 256. (b) Ablation study of our method without finetuning appearance reference module, with unaligned reference images, and with aligned reference images, evaluated on NeRSemble at the resolution of 512 _×_ 512. 

thesis _I_[˜] at the target novel view, which is typically blurry but 3D consistent. We then diffuse the latent feature _E_ ( _I_[˜] ) with 1000 time steps into a Gaussian noise as the input to the LDM. In essence, the two-step generated noise still contains some image layout semantics in a very coarse grain and in practice, enhanced view consistency is observed in our task as demonstrated in Figure 5. 

## **4. Experiments** 

**Dataset and Training.** Our model was trained in three stages on our multi-view image dataset as an image reconstruction task. That being said, both the appearance reference image _Iref_ and the target image _IT_ are sourced from the same identity but with different views, whereas _Icam_ is synthesized with EG3D [6] using a random latent Gaussian noise and the calibrated camera parameters of _IT_ . We lock the parameters of the SD-UNet _F_ during the whole training stage. In the first stage, we train all the parameters of our appearance reference module _Fref_ without any camera guidance. Next we freeze the weights of _Fref ,_ and train our view control module _Fcam_ with paired _Icam._ Lastly the view consistency module, performing cross-view attentions among 8 views at once, is trained with the rest modules frozen. All training was conducted on 6 Nvidia A100 GPUs at a learning rate of 10 _[−]_[5] , with 16 images processed in each step. During inference, we empirically set 100 steps for DDIM denoising [44] and unconditional guidance scale [19] as 3 for a good balance of quality and speed. 

We trained our modules on a hybrid dataset comprised of photo-realistic multi-view images NeRSemble [29] and synthetic ones by PanoHead [2]. NeRSemble dataset consists of high-resolution videos of 220 subjects performing a wide range of dynamic expressions, captured from 16 calibrated synchronized cameras. We sampled 2000 pairs of multi-view frames from NeRSemble for training, where 1 randomly-selected view is used for appearance reference and 8 other views as targets. Given the scarcity of available 

camera views and the background variation, we augmented our training dataset with another 2000 pairs of multi-view images synthesized via PanoHead [2]. For evaluation, we used another unseen 500 multi-view pairs from NeRSemble, and 360 single-view internet-collected in-the-wild portraits, containing a wide variation in appearance, expression, camera perspective, and style. We note that for training, all the images are cropped and aligned as in EG3D [6] whereas we do not perform image alignment during inference (unless explicitly stated for comparison to GAN-based methods). For testing on both datasets, the novel camera views are all manipulated with EG3D renderings. 

## **4.1. Qualitative Evaluations** 

Given a single reference portrait, our method demonstrates high-fidelity and 3D-consistent novel view synthesis at a resolution of 512 _×_ 512 _,_ as illustrated in Figure 1. While only being trained on aligned real portrait images, our method shows superior generalization capability to novel identities, styles, expressions and views. This is largely credited to the preservation of the generative prior of pretrained LDM by our design. As evidenced in Figure 4, our view control module is also able to effectively control the synthesis view. Compared to the ground truth (second column, Figure 4), our novel portraits are highly plausible but with some noticeable identity differences. This is due to the limited visual appearance clue in the single reference image, and the problem can be largely alleviated with additional references (please refer to Figure 8 and the supplementary paper for visual results). 

We extensively compare to a few state-of-the-art novel portrait synthesis works on both image reconstruction (Figure 6) and novel view synthesis (Figure 3, 4): GOAE [57],TriPlaneNet [3], Pivot Tuning (EG3D-PTI) [6] and Zero-1-to-3 [34]. GOAE [57] and TriPlaneNet [3] designed an effective image encoder for EG3D [6], whereas EG3D-PTI runs latent code optimization and finetunes the 

7 

weights of EG3D per image. We did not compare to Live3D Portrait [47] given unavailable implementation and model. Zero-1-to-3 [34] leverages Stable Diffusion but was trained on 3D object dataset Objaverse [9]. While not required by our method, we cropped and aligned the test images as in EG3D. Nevertheless, our method outperforms substantially over the prior work in terms of both perceptual quality, and preservation of identity and expression. Notably all 3D GAN-based baselines fail to reconstruct side views (Figure 6), exaggerated expressions (Figure 4), or out-ofdomain styles (Figure 3), whereas Zero-1-to-3 synthesizes novel portraits with very limited perceptual quality. 

## **4.2. Quantitative Evaluations** 

We evaluate methods for single-view novel portrait synthesis on 4 main aspects. We use LPIPS _↓_ [60], DISTS _↓_ [14], SSIM _↑_ [50] for evaluation of 2D image reconstruction, ID _↑_ [10] for identity consistency, FID _↓_ [18] for perceptual quality, and POSE _↓_ for camera view control accuracy. Notably, to evaluate reconstruction fairly, we estimate camera parameters from the ground-truth target image and uses the EG3D renderings as condition _Icam._ The error in camera estimation could result in some image misalignment and therefore we mainly rely on perceptual metrics LPIPS and DISTS for reconstruction evaluation. The identity similarity is calculated between the synthesized and reference image by calculating the cosine similarity of the face embeddings with a pretrained face recognition module [10]. 

Table 1a shows the numerical comparison on reconstruction of NeRSemble and in-the-wild test images, and novel view synthesis of NeRSemble respectively. On all image metrics, our method shows our method is superior than all prior work by a large margin, demonstrating the most compelling image quality. Our pose reconstruction is slightly worse than the baseline. However, we argue that this is largely due to the camera misalignment between the ground truth and the condition EG3D rendering. 

## **4.3. Ablations** 

We ablate the efficacy of the individual component with extensive ablation experiments for noval view synthesis on NeRSemble test set. As illustrated in Figure 5, we demonstrate the necessity of our view consistency module and 3D-aware noise in maintaining appearance coherence cross multiple views. Without them, substantial variations are observed, especially on the unobserved region of the reference image, when altering the camera views. The weights of our appearance reference module is initiated from a copy of SDUNet which should be already able to derive local appearance context from the reference image. However, as evidenced by Table 1b and Figure 7, significant improvements are achieved by our finetuning on multi-view images. We reason that the necessity of finetuning is due to the removal 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0008-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0008-07.png)


**----- Start of picture text -----**<br>
reference ground truth w/o fine-tuning ours<br>appearance reference<br>**----- End of picture text -----**<br>


Figure 7. Fine-tuning appearance reference module helps better retain the spatial features from the reference image. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0008-09.png)


**----- Start of picture text -----**<br>
view 1 view 2 view 3<br>use view 1 use view 1,2 use view 1,2,3 ground truth<br>reference<br>novel view<br>**----- End of picture text -----**<br>


Figure 8. Our method seamlessly supports multiple reference images as input, and the novel view synthesis quality is progressively enhanced with more references. 

of cross attention from text. Lastly unlike many GAN-based methods that requires the reference image to be aligned, our model supports free-form portraits as inputs without quality degeneration even though the model was trained on cameraaligned multi-view images. This is numerically shown in Table 1b where aligning the reference image (as in EG3D) only leads to neglectable differences. 

## **5. Discussion** 

**Conclusion.** We presented _DiffPortrait3D_ , a novel conditional diffusion model that is capable of generating consistent novel portraits from sparse input views. By design, our framework seamlessly cross-references the key characteristics from the input images and effectively adds camera pose control into the latent diffusion process, modulated with enhanced consistency across views. Trained only with a few thousand of synthetic and real multi-view images, our model successfully showcases compelling novel portrait synthesis results, regardless of appearances, expressions, camera perspectives, and styles. This is largely credited to our explicitly disentangled control of appearance and view within both model design and training, without harming the generalization capability of large pretrained diffusion models. We believe that our framework opens up possibilities for accessible 3D reconstruction and visualization from a single picture. 

8 

## **DiffPortrait3D: Controllable Diffusion for Zero-Shot Portrait View Synthesis** 

## Supplementary Material 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0009-02.png)


Figure 9. shows how our 3D convolution-based novel view synthesis pipeline _S_ works. In practice, a 3D-convolution-based network first maps the reference image into a 3D feature volume. Then, given a conditioned camera view, we follow the volume rendering to integrate the 3D features into a 2D feature map which is further decoded to the final RGB image _I_[˜] with a 2D convolution network. During the inference phase, we enhance 3D awareness by commencing with noise generated via a 1000-step forward diffusion process applied to _E_ ( _I_[˜] ), which serves as the initial noise for our DiffPortrait3D pipeline. 

In this supplementary paper, we provide additional implementation details in Section A, showcase more visual results and numerical comparisons in Section B, and discuss limitations & ethics consideration in Section C and Section D . 

## **A. Implementation Detail** 

## **A.1. 3D-Aware Noise** 

In Figure 9 and Figure 10, we illustrate the framework of generating our “3D-aware” noise. Specifically, we build a 3D convolution-based novel view synthesis pipeline (denoted as _S_ ), trained as a multi-view image reconstruction task. Similar to [49] and [42], we first employ a 3D appearance feature extraction network to map the reference image to a 3D appearance feature volume. To synthesize an image at a novel view, we follow the volume rendering as in NeRF [35] to integrate the 3D features into a 2D feature map which is further decoded to the final RGB image _I_[˜] with a deep 2D convolutional network. The network modules are trained with image reconstruction losses against ground-truth multi-view images, including pixel-aligned L2 and VGG perceptual losses [24]. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0009-08.png)


Figure 10. Our 3D-Aware Noise effectively helps strengthen the novel view synthesis result. 

During inference, given a reference image, we first employ our trained 3D novel view synthesis network _S_ to generate a proxy rendering _I_[˜] at the target view. While being blurry, _I_[˜] contains rich 3D structural semantics and acts as a good guidance to the diffusion process in our DiffPortrait3D. We incorporate this 3D awareness by generating the starting noise using the forward noising process of 1000 steps applied to the latent map of _I,_[˜] i.e., _E_ ( _I_[˜] ) _._ Better reconstruction and consistency are observed with our proposed 3D-aware noise, as evidenced in Table 2 numerically and in Figure 5 of the main paper visually. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0009-11.png)


Figure 11. Ablation on view conditional images. 

## **A.2. Metrics** 

## **A.2.1 Identity Similarity** 

Our identity similarity score (ID) is calculated based on the cosine similarity of the face embeddings with a pre-trained face recognition module[10] as , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0009-16.png)


1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0010-00.png)


Figure 12. Novel view synthesis of anthropomorphic animals. 

where _fg′_ , and _fg_ are the feature embedding of the generated image and ground-truth image respectively. 

## **A.2.2 Pose Accuracy** 

We evaluate the pose accuracy (POSE) with the assistance of an off-the-shelf face reconstruction model [11]. We detect pitch, yaw, and roll from the generated novel view images, then compute the L2 loss against the camera poses estimated from ground truth images. 

## **A.3. Baselines** 

## **A.4. EG3D-Pivot Tuning Inversion** 

For our baseline EG3D-PTI, we follow the standard procedure as described in [39], where for each reference image, we first optimize the latent noise for 500 iterations and further fine-tune the generator weights for another additional 250 iterations. Once completed, we used the optimized latent noise finetuned 3D-aware generator to synthesize the image at novel views. 

## **A.5. Zero-1-to-3** 

Zero-1-to-3 [34] is one of the state-of-the-art novel diffusion-based view synthesis works designed for general 3D objects. Nevertheless, we compare to it for a thorough evaluation of existing works on novel portrait synthesis. In Table 3, to maximize its performance on our task, our numerical results are all based on portraits with removed backgrounds. Additionally, due to the differences in 3D camera coordinates, we only report the FID and identity similarity (ID) of the novel view synthesis results for a fair comparison. 

## **B. More Experiment results** 

## **B.1. Alignments** 

Our model was trained with EG3D-aligned reference and target images. However, our method does not restrict the reference images to be cropped and aligned, nor with the camera condition images. In Figure 13, we showcase that 

||3D-Aware noise|Random noise|
|---|---|---|
|LPIPS_↓_|**0.27**|0.32|
|SSIM_↑_|**0.68**|0.65|
|DIST_↓_|**0.18**|0.21|
|ID_↑_|**0.70**|**0.70**|
|FID_↓_|**25.37**|26.81|



Table 2. Quantitative ablation of 3D-Aware Noise and Random Noise results of novel view synthesis of NeRSemble [29] at the resolution of 512x512 

||Ours|Zero1to3|
|---|---|---|
|LPIPS_↓_|**0.04/0.19/0.04**|0.28/-/0.34|
|SSIM_↑_|**0.90/0.74/0.88**|0.70/-/0.52|
|DIST_↓_|**0.05/0.15/0.04**|0.16/-/0.19|
|ID_↑_|**0.94/0.70/0.92**|0.82/0.09/0.70|
|FID_↓_|**6.5/32.6/11.6**|61.4/113.8/57.5|



Table 3. Quantitative comparison of our method and Zero-1-to-3, showing numerical results of reconstruction/novel view synthesis of NeRSemble [29], and reconstruction of in-the-wild test images( from left to right). For a fair comparison to Zero-1-to-3, the evaluation is performed with the removed backgrounds at the resolution of 512 _×_ 512. 

with differently aligned reference images, our method synthesizes close novel view results. 

## **B.2. PanoHead-PTI Comparison** 

PanoHead extends the EG3D framework by enabling novel view synthesis in 360 _[◦]_ . However, owing to the inherent limitations of GAN-based architecture, PanoHead, like EG3D, necessitates time-consuming instance-specific optimization (pivot-tuning) while still suffers from limited perceptual quality and identity loss, especially for portraits with out-ofdomain styles or extreme expressions (as shown in Fig. 14 compared to our results in Fig. 4 of the main paper). Our method is superior by a large margin on quantitative metrics as well (POSE _↓_ -/0.0023/-, LPIPS _↓_ 0.22/0.28/0.11, SSIM _↑_ 0.60/0.53/0.76, DIST _↓_ 0.18/0.26/0.12, ID _↑_ 0.47/0.38/0.12, FID _↓_ 56.53/60.4/90.47; ours are detailed in Tab.1 of the main paper). 

## **B.3. View-consistent novel view synthesis** 

We show more challenging results in Figure 16 , 17 and 18. Our model is able to generalize well to arbitrary face portraits with unposed camera views, extreme facial expressions, and diverse artistic depictions. Please also refer to 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0011-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0011-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0011-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0011-03.png)


**----- Start of picture text -----**<br>
aligned reference  ours unaligned reference ours<br>**----- End of picture text -----**<br>


Figure 13. DiffPortrait3D effectively derives appearance features from the reference image, without strict restriction to its image alignment. Similar novel view synthesis results are achieved using EG3D-aligned and non-aligned reference images. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0011-05.png)


Figure 14. Novel view synthesis with PanoHead-PTI. 

our supplementary video for more high-resolution results. 

## **B.4. Ablation on view condition images** 

Our method effectively disentangles the control of camera views from appearance. As shown in Fig. 11, visual differences are hardly noticeable between the synthesized novel views when using two view conditions generated at the same camera pose but with distinct appearance seeds. For quantitative assessment, we perform novel view synthesis across all our test images using two sets of view conditional images generated under the same camera pose 

but featuring different appearances. We calculate the differences in image pixels (LPIPS _↓_ 0.09) and camera poses (POSE _↓_ 0.0041). Note that the LPIPS difference is partially attributed to slight structural shifting. 

## **B.5. Anthropomorphic animals** 

While being trained sorely on real human images, our method is empowered with strong domain generalization capability (e.g., Fig. 12) by leveraging the generative prior of a pre-trained stable diffusion model. However, we acknowledge that visual artifacts are possible due to the appearance bias originating from the training data distribution. 

## **C. Limitation and Future Work** 

While the image coherence is largely strengthened with our cross-view module and 3D-aware generation, we still observe occasionally flickering artifacts in unobserved re- 

3 

gions. We leave the exploration of longer-range consistent view manipulation as future work. In this work, the appearance is formulated to be sourced from the reference images only. This could result in some loss of identity given the limited appearance context. In the future, we would like to extend our framework such that the identity can be multisourced from e.g., text and personalized Loras [23]. As discussed above, we also include visualizations of failure cases in Figure 15. Rows (a) through (f) display artifacts in areas not observed, accompanied by changes in identity, particularly noticeable in (b), (c), and (d). In cases (a), (e), and (f), it is evident that the model struggles to accurately replicate secondary elements from the reference image, such as sunglasses in (f), hands and flowers in (e), and leaves in (a), leading to inconsistent outcomes. One potential issue we’ve identified stems from the use of a 2D diffusion backbone that integrates 3D-Aware information. This approach, while innovative, may lead to minor inconsistencies and deviations, especially in challenging depictions. Addressing these limitations is an important area that should be addressed in future work. 

## **D. Ethic Consideration** 

We acknowledge the profound capabilities of the diffusion model as a powerful generative model. The framework proposed in our paper could, theoretically, be utilized to compromise multi-perspective facial recognition systems. We assert that the model and the accompanying research code are intended exclusively for advancing scientific research and must not be used for illicit purposes. 

## **References** 

- [1] Stability AI. Stable diffusion v1.5 model card. _https://huggingface.co/runwayml/stable-diffusion-v1-5_ , 2022. 2, 4 

- [2] Sizhe An, Hongyi Xu, Yichun Shi, Guoxian Song, Umit Y. Ogras, and Linjie Luo. Panohead: Geometry-aware 3d fullhead synthesis in 360deg. In _CVPR_ , pages 20950–20959, 2023. 1, 2, 3, 6, 7 

- [3] Ananta R Bhattarai, Matthias Nießner, and Artem Sevastopolsky. Triplanenet: An encoder for eg3d inversion. _arXiv preprint arXiv:2303.13497_ , 2023. 2, 3, 7 

- [4] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. _arXiv preprint arXiv:2304.08465_ , 2023. 4 

- [5] Eric R Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, and Gordon Wetzstein. pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 5799–5809, 2021. 1, 3 

- [6] Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas Guibas, Jonathan Tremblay, Sameh Khamis, Tero 

   - Karras, and Gordon Wetzstein. Efficient geometry-aware 3D generative adversarial networks. In _CVPR_ , 2022. 1, 3, 6, 7 

- [7] Eric R. Chan, Koki Nagano, Matthew A. Chan, Alexander W. Bergman, Jeong Joon Park, Axel Levy, Miika Aittala, Shalini De Mello, Tero Karras, and Gordon Wetzstein. GeNVS: Generative novel view synthesis with 3D-aware diffusion models. In _arXiv_ , 2023. 3 

- [8] Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, and Pieter Abbeel. Infogan: Interpretable representation learning by information maximizing generative adversarial nets, 2016. 2 

- [9] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 13142–13153, 2023. 8 

- [10] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In _CVPR_ , pages 4690–4699, 2019. 8, 1 

- [11] Yu Deng, Jiaolong Yang, Sicheng Xu, Dong Chen, Yunde Jia, and Xin Tong. Accurate 3d face reconstruction with weakly-supervised learning: From single image to image set. In _IEEE Computer Vision and Pattern Recognition Workshops_ , 2019. 2 

- [12] Yu Deng, Jiaolong Yang, Jianfeng Xiang, and Xin Tong. Gram: Generative radiance manifolds for 3d-aware image generation. _CVPR_ , 2022. 1, 3 

- [13] Abdallah Dib, Cedric Thebault, Junghyun Ahn, PhilippeHenri Gosselin, Christian Theobalt, and Louis Chevallier. Towards high fidelity monocular face reconstruction with rich reflectance using self-supervised learning and ray tracing, 2021. 2 

- [14] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P. Simoncelli. Image quality assessment: Unifying structure and texture similarity. _CoRR_ , abs/2004.07728, 2020. 8 

- [15] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In _NeurIPS_ , 2014. 2 

- [16] Jiatao Gu, Lingjie Liu, Peng Wang, and Christian Theobalt. Stylenerf: A style-based 3d-aware generator for highresolution image synthesis. _CVPR_ , 2022. 1, 2, 3 

- [17] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. _arXiv preprint arXiv:2307.04725_ , 2023. 2, 6 

- [18] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In _Advances in Neural Information Processing Systems_ . Curran Associates, Inc., 2017. 8 

- [19] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. _arXiv preprint arXiv:2207.12598_ , 2022. 7 

- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. _NeurIPS_ , 33:6840–6851, 2020. 2, 4 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0013-00.png)


**----- Start of picture text -----**<br>
���<br>���<br>���<br>���<br>���<br>���<br>**----- End of picture text -----**<br>


Figure 15. Failure Case 

- [21] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video dif- 

fusion models. _arXiv:2204.03458_ , 2022. 2 

- [22] Yang Hong, Bo Peng, Haiyao Xiao, Ligang Liu, and Juyong 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0014-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0014-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0014-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0014-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0014-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0014-05.png)


Figure 16. More Novel-view consistent results 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0015-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0015-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0015-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0015-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0015-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0015-05.png)


Figure 17. More Novel-view consistent results 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0016-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0016-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0016-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0016-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0016-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis_images/cvpr_2024_DiffPortrait3D_Controllable_Diffusion_for_Zero-Shot_Portrait_View_Synthesis.pdf-0016-05.png)


Figure 18. More Novel-view consistent results 

8 

Zhang. Headnerf: A real-time nerf-based parametric head model. In _CVPR_ , pages 20374–20384, 2022. 2 

- [23] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. _arXiv preprint arXiv:2106.09685_ , 2021. 4 

- [24] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In _Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14_ , pages 694–711. Springer, 2016. 1 

- [25] Omer Kafri, Or Patashnik, Yuval Alaluf, and Daniel CohenOr. Stylefusion: A generative model for disentangling spatial segments, 2021. 2 

- [26] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In _CVPR_ , 2019. 2 

- [27] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of StyleGAN. In _CVPR_ , 2020. 2 

- [28] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In _CVPR_ , 2020. 3 

- [29] Tobias Kirschstein, Shenhan Qian, Simon Giebenhain, Tim Walter, and Matthias Nießner. Nersemble: Multi-view radiance field reconstruction of human heads. _ACM Transactions on Graphics_ , 2023. 2, 5, 7 

- [30] Dongxu Li, Junnan Li, and Steven CH Hoi. Blipdiffusion: Pre-trained subject representation for controllable text-to-image generation and editing. _arXiv preprint arXiv:2305.14720_ , 2023. 4 

- [31] Connor Z Lin, David B Lindell, Eric R Chan, and Gordon Wetzstein. 3d gan inversion for controllable portrait image animation. _arXiv preprint arXiv:2203.13441_ , 2022. 2, 3 

- [32] Yukang Lin, Haonan Han, Chaoqun Gong, Zunnan Xu, Yachao Zhang, and Xiu Li. Consistent123: One image to highly consistent 3d asset using case-aware diffusion priors, 2023. 3 

- [33] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, and Hao Su. One-2-3-45: Any single image to 3d mesh in 45 seconds without pershape optimization, 2023. 3 

- [34] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object, 2023. 3, 7, 8, 2 

- [35] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In _ECCV_ , 2020. 2, 1 

- [36] Roy Or-El, Xuan Luo, Mengyi Shan, Eli Shechtman, Jeong Joon Park, and Ira Kemelmacher-Shlizerman. Stylesdf: High-resolution 3d-consistent image and geometry generation. _CVPR_ , 2022. 1, 3 

- [37] Pascal Paysan, Reinhard Knothe, Brian Amberg, Sami Romdhani, and Thomas Vetter. A 3d face model for pose and illumination invariant face recognition. In _2009 sixth IEEE international conference on advanced video and signal based surveillance_ , pages 296–301. Ieee, 2009. 2 

- [38] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. _arXiv_ , 2022. 3 

- [39] Daniel Roich, Ron Mokady, Amit H Bermano, and Daniel Cohen-Or. Pivotal tuning for latent-based editing of real images. _ACM Transactions on Graphics_ , 42(1):1–13, 2022. 2, 3 

- [40] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In _CVPR_ , pages 10684– 10695, 2022. 2, 4 

- [41] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. _NeurIPS_ , 35:36479–36494, 2022. 2 

- [42] Aliaksandr Siarohin, St´ephane Lathuili`ere, Sergey Tulyakov, Elisa Ricci, and Nicu Sebe. First order motion model for image animation. In _Conference on Neural Information Processing Systems (NeurIPS)_ , 2019. 1 

- [43] Ivan Skorokhodov, Sergey Tulyakov, Yiqun Wang, and Peter Wonka. Epigraf: Rethinking training of 3d gans. _arXiv preprint arXiv:2206.10535_ , 2022. 1, 3 

- [44] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. _arXiv preprint arXiv:2010.02502_ , 2020. 2, 4, 7 

- [45] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. _arXiv preprint arXiv:2011.13456_ , 2020. 2, 4 

- [46] Alex Trevithick, Matthew Chan, Michael Stengel, Eric R. Chan, Chao Liu, Zhiding Yu, Sameh Khamis, Manmohan Chandraker, Ravi Ramamoorthi, and Koki Nagano. Realtime radiance fields for single-image portrait view synthesis. In _SIGGRAPH_ , 2023. 2, 3 

- [47] Alex Trevithick, Matthew Chan, Michael Stengel, Eric R. Chan, Chao Liu, Zhiding Yu, Sameh Khamis, Manmohan Chandraker, Ravi Ramamoorthi, and Koki Nagano. Realtime radiance fields for single-image portrait view synthesis, 2023. 8 

- [48] Satoshi Tsutsui, Weijia Mao, Sijing Lin, Yunyi Zhu, Murong Ma, and Mike Zheng Shou. Novel view synthesis for highfidelity headshot scenes. _arXiv preprint arXiv:2205.15595_ , 2022. 2 

- [49] Ting-Chun Wang, Arun Mallya, and Ming-Yu Liu. One-shot free-view neural talking-head synthesis for video conferencing. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , 2021. 1 

- [50] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: From error visibility to structural similarity. _IEEE Transactions on Image Processing_ , 13(4):600–612, 2004. 8 

- [51] Haohan Weng, Tianyu Yang, Jianan Wang, Yu Li, Tong Zhang, CL Chen, and Lei Zhang. Consistent123: Improve consistency for one image to 3d object synthesis. _arXiv preprint arXiv:2310.08092_ , 2023. 4 

- [52] Fanzi Wu, Linchao Bao, Yajing Chen, Yonggen Ling, Yibing Song, Songnan Li, King Ngi Ngan, and Wei Liu. Mvf-net: 

9 

Multi-view 3d face morphable model regression. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 959–968, 2019. 2 

- [53] Sitao Xiang, Yuming Gu, Pengda Xiang, Mingming He, Koki Nagano, Haiwei Chen, and Hao Li. One-shot identity-preserving portrait reenactment. _arXiv preprint arXiv:2004.12452_ , 2020. 2 

- [54] Yiheng Xie, Towaki Takikawa, Shunsuke Saito, Or Litany, Shiqin Yan, Numair Khan, Federico Tombari, James Tompkin, Vincent Sitzmann, and Srinath Sridhar. Neural fields in visual computing and beyond. In _Computer Graphics Forum_ , pages 641–676. Wiley Online Library, 2022. 2 

- [55] Xudong Xu, Xingang Pan, Dahua Lin, and Bo Dai. Generative occupancy fields for 3d surface-aware image synthesis. _NeurIPS_ , 34:20683–20695, 2021. 1, 3 

- [56] Yang Xue, Yuheng Li, Krishna Kumar Singh, and Yong Jae Lee. Giraffe hd: A high-resolution 3d-aware generative model. In _CVPR_ , 2022. 1, 3 

- [57] Ziyang Yuan, Yiming Zhu, Yu Li, Hongyu Liu, and Chun Yuan. Make encoder great again in 3d gan inversion through geometry and occlusion-aware encoding. _arXiv preprint arXiv:2303.12326_ , 2023. 2, 3, 7 

- [58] Jingbo Zhang, Xiaoyu Li, Ziyu Wan, Can Wang, and Jing Liao. Fdnerf: Few-shot dynamic neural radiance fields for face reconstruction and expression editing, 2022. 2 

- [59] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In _ICCV_ , pages 3836–3847, 2023. 2, 4, 5 

- [60] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 586–595, 2018. 8 

- [61] Yiyu Zhuang, Hao Zhu, Xusen Sun, and Xun Cao. Mofanerf: Morphable facial neural radiance field. In _European Conference on Computer Vision_ , pages 268–285. Springer, 2022. 2 

10 

