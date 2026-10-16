# **Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning** 

Antoine Yang _[†]_[*] Arsha Nagrani _[§]_ Paul Hongsuck Seo _[§]_ Antoine Miech _[♯]_ Jordi Pont-Tuset _[§]_ Ivan Laptev _[†]_ Josef Sivic _[¶]_ Cordelia Schmid _[§]_ 

> _§_ Google Research _†_ Inria Paris and D´epartement d’informatique de l’ENS, CNRS, PSL Research University 

> _♯_ DeepMind _¶_ Czech Institute of Informatics, Robotics and Cybernetics at the Czech Technical University in Prague https://antoyang.github.io/vid2seq.html 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Vid2Seq_Large-Scale_Pretraining_of_a_Visual_Language_Model_for_Dense_Video_Captioning_images/cvpr_2023_Vid2Seq_Large-Scale_Pretraining_of_a_Visual_Language_Model_for_Dense_Video_Captioning.pdf-0001-05.png)


**----- Start of picture text -----**<br>
Large-scale pretraining from narrated videos<br>Vid2Seq<br>Hey guys today I am going<br>to teach you how to ski The kids make it look easy First slope, congratz!<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Vid2Seq_Large-Scale_Pretraining_of_a_Visual_Language_Model_for_Dense_Video_Captioning_images/cvpr_2023_Vid2Seq_Large-Scale_Pretraining_of_a_Visual_Language_Model_for_Dense_Video_Captioning.pdf-0001-06.png)


**----- Start of picture text -----**<br>
Dense video captioning<br><1s><8s>The man is fastening the dog. <20s><50s>The dogs are pulling the sled. <45s><49s>The man is saying hello.<br>**----- End of picture text -----**<br>


Figure 1. **Vid2Seq** is a visual language model that predicts dense event captions together with their temporal grounding in the video by generating a _single_ sequence of tokens (right). This ability is enabled by large-scale pretraining on unlabeled narrated videos (left). 

## **Abstract** 

_In this work, we introduce Vid2Seq, a multi-modal single-stage dense event captioning model pretrained on narrated videos which are readily-available at scale. The Vid2Seq architecture augments a language model with special time tokens, allowing it to seamlessly predict event boundaries and textual descriptions in the same output sequence. Such a unified model requires large-scale training data, which is not available in current annotated datasets. We show that it is possible to leverage unlabeled narrated videos for dense video captioning, by reformulating sentence boundaries of transcribed speech as pseudo event boundaries, and using the transcribed speech sentences as pseudo event captions. The resulting Vid2Seq model pretrained on the YT-Temporal-1B dataset improves the state of the art on a variety of dense video captioning benchmarks including YouCook2, ViTT and ActivityNet Captions. Vid2Seq also generalizes well to the tasks of video paragraph captioning and video clip captioning, and to few-shot settings. Our code is publicly available at [1]._ 

## **1. Introduction** 

Dense video captioning requires the temporal localization and captioning of all events in an untrimmed video [46, 102, 131]. This differs from standard video captioning [63, 70, 80], where the goal is to produce a single caption for a given short video clip. Dense captioning is significantly more difficult, as it raises the additional complexity of localizing the events in minutes-long videos. However, it also benefits from long-range video information. This task is 

> *This work was done when the first author was an intern at Google. 

potentially highly useful in applications such as large-scale video search and indexing, where the video content is not segmented into clips. 

Existing methods mostly resort to two-stage approaches [37, 46, 100], where events are first localized and then captioned. To further enhance the inter-task interaction between event localization and captioning, some approaches have introduced models that jointly solve the two tasks [20, 102, 131]. However, often these approaches still require task-specific components such as event counters [102]. Furthermore, they exclusively train on manually annotated datasets of limited size [35, 46, 130], which makes it difficult to effectively solve the task. To address these issues, we take inspiration from recent sequenceto-sequence models pretrained on Web data which have been successful on a wide range of vision and language tasks [4, 11, 13, 105, 117]. 

First, we propose a video language model, called Vid2Seq. We start from a language model trained on Web text [78] and augment it with special _time tokens_ that represent timestamps in the video. Given video frames and transcribed speech inputs, the resulting model jointly predicts all event captions and their corresponding temporal boundaries by generating a _single_ sequence of discrete tokens, as illustrated in Figure 1 (right). Such a model therefore has the potential to learn multi-modal dependencies between the different events in the video via attention [94]. However this requires large-scale training data, which is not available in current dense video captioning datasets [35,46,130]. Moreover, collecting manual annotations of dense captions for videos is expensive and prohibitive at scale. 

1 

Hence we propose to pretrain Vid2Seq by leveraging unlabeled narrated videos which are readily-available at scale. To do this, we reformulate sentence boundaries of transcribed speech as pseudo event boundaries, and use the transcribed speech sentences as pseudo event captions. We then pretrain Vid2Seq with a generative objective, that requires predicting the transcribed speech given visual inputs, and a denoising objective, which masks spans of transcribed speech. Note that transcribed speech may not describe the video content faithfully, and is often temporally misaligned with the visual stream [32, 43, 71]. For instance, from the example in Figure 1 (left), one can understand that the grey skier has descended a slope from the last speech sentence which is said _after_ he actually descended the slope. Intuitively, Vid2Seq is particularly suited for learning from such noisy supervision as it jointly models _all_ narrations and the corresponding timestamps in the video. 

We demonstrate the effectiveness of our pretrained model through extensive experiments. We show the importance of pretraining on untrimmed narrated videos, the ability of Vid2Seq to use both the visual and speech modalities, the importance of the pretraining objectives, the benefit of joint caption generation and localization, as well as the importance of the language model size and the scale of the pretraining dataset. The pretrained Vid2Seq model achieves state-of-the-art performance on various dense video captioning benchmarks [35, 46, 130]. Our model also excels at generating paragraphs of text describing the video: without using ground-truth event proposals at inference time, our model outperforms all prior approaches including those that rely on such proposals [50,76,128]. Moreover, Vid2Seq generalizes well to the standard task of video clip captioning [9, 109]. Finally, we introduce a new few-shot dense video captioning setting in which we finetune our pretrained model on a small fraction of the downstream training dataset and show benefits of Vid2Seq in this setting. 

In summary, we make the following contributions: _(i)_ We introduce Vid2Seq for dense video captioning. Given multi-modal inputs (transcribed speech and video), Vid2Seq predicts a single sequence of discrete tokens that includes caption tokens interleaved with special _time tokens_ that represent event timestamps. _(ii)_ We show that transcribed speech and corresponding timestamps in unlabeled narrated videos can be effectively used as a source of weak supervision for dense video captioning. _(iii)_ Finally, our pretrained Vid2Seq model improves the state of the art on three dense video captioning datasets (YouCook2, ViTT, ActivityNet Captions), two video paragraph captioning benchmarks (YouCook2, ActivityNet Captions) and two video clip captioning datasets (MSR-VTT, MSVD), and also generalizes well to few-shot settings. 

Our code implemented in Jax and based on the Scenic library [19] is publicly released at [1]. 

## **2. Related Work** 

**Dense video captioning.** Dense video captioning lies at the intersection of event localization [25, 29, 33, 62, 65, 66, 84, 127] and event captioning [30,63,75,97,104]. The majority of existing methods for dense video captioning [37, 38, 46, 100, 103] consist of a temporal localization stage followed by an event captioning stage. To enrich inter-task interactions, recent works [8,10,20,61,73,79,82,83,100,102,131] jointly train the captioning and localization modules. In particular, Wang _et al_ . [102] propose to view dense video captioning as a set prediction task, and jointly perform event localization and captioning for each event in parallel. In contrast, our model generates event boundaries and captions conditioned on the previously generated events. Deng _et al_ . [20] propose to first generate a paragraph and then ground each sentence in the video. We also generate all captions as a single output sequence, however our output already includes event timestamps. Zhang _et al_ . [125] propose to generate event boundaries sequentially, but separately perform event localization and single event captioning, and only use visual input. Most related to our work, Zhu _et al_ . [133] also perform dense video captioning by generating a single output sequence. Their method, however, infers event locations directly from the timestamps of transcribed speech and, hence, can only detect events that closely follow the speech. In contrast, our model generates event timestamps as special tokens and can produce dense captions for videos with limited speech, as we demonstrate on the ActivityNet Captions dataset. 

**Video and language pretraining.** Following the success of image-text pretraining [14, 21, 23, 24, 28, 34, 36, 39–41, 53– 55,58,60,68,69,85,88,92,93,99,118–120,124,129], recent works have explored video-text pretraining [3–5, 26, 31, 32, 43, 49, 52, 56, 71, 72, 74, 80, 81, 89, 96, 98, 108, 111, 111–115, 121, 122]. These methods show strong improvements on various tasks such as text-video retrieval [5,71], video question answering [112, 122] and video clip captioning [4, 80]. While these works mostly learn global video representations to tackle video-level prediction tasks, we here focus on learning detailed representations to address a dense prediction task requiring reasoning over multiple events in untrimmed videos. Several works have explored long-form video-text pretraining [90] and video-text pretraining for temporal localization tasks [7, 48, 64, 106, 110, 116]. However these works focus on video understanding tasks while our pretraining approach is tailored for a generative task that not only requires the model to reason over multiple events in the video, but also to describe them by natural language. 

A few works explore pretraining for dense video captioning. Zhang _et al_ . [125] pretrain on ActivityNet Captions to improve the downstream performance on the same dataset. In contrast, we propose a pretraining method that does not rely on _any_ manual annotation, and show its ben- 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Vid2Seq_Large-Scale_Pretraining_of_a_Visual_Language_Model_for_Dense_Video_Captioning_images/cvpr_2023_Vid2Seq_Large-Scale_Pretraining_of_a_Visual_Language_Model_for_Dense_Video_Captioning.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Visual token Output dense  0.50s 20.08s àà 8.53s:  49.70s: The man is fastening the dog. The dogs are pulling the sled.<br>Time token event captions 44.68s à 49.20s: The man is saying hello.<br>Text token<br>Visual and speech embeddings [ x [t] , y [t] ] Output event sequence z<br>x1 [t] x2 [t] xF-1 [t] xF [t] y1 [t] y2 [t] y3 [t] y4 [t] y5 [t] y6 [t] y7 [t] y8 [t] y9 [t] ys [t] <1>    <17>    The      man     is fastening … <98>     The      man     is saying hello. [EOS]<br>...<br>Language Modeling Head  h [l]<br>Temporal Encoder   f  [t] Transformer Text Encoder   g [t] Transformer Text Decoder h [t]<br>x1 [s] x2 [s] ...  xF-1 [s] xF [s] y1 [s] y2 [s] y3 [s] y4 [s] y5 [s] y6 [s] y7 [s] y8 [s] y9 [s] ys [s] z1 [s] z2 [s] z3 [s] z4 [s] z5 [s] [z] 6 [s] ...  zL-6 [s] zL-5 [s] zL-4 [s] [z] L-3 [s] [z] L-2 [s] zL-1 [s] zL [s]<br>Spatial Encoder  f  [s] Encoder Token Embedder g [s] Decoder Token Embedder  h [s]<br>...<br><6>  <10>    Please stay calm!    <86>    <92>    Hey     my friend! [BOS]    <1>      <17>     The    man    is <89>     <98>      The    man    is saying hello.<br>Transcribed speech sequence<br>Text + Time  Tokens<br>Text + Time Tokenization stay calm man          is… hey  <0> <1>       <99>…<br>Time Tokenization<br>3.02s   42.87s àà 4.99s:  45.97s: Please stay calm!Hey my friend!  s1=3.02s  tstart1 = s1#×" = $.&'×)*.+&(&& = 6<br>Input video frames   x Input transcribed speech Video timeline, duration T = 49.70s quantized in N = 100 bins<br>f g<br>h<br>Visual Encoder   Text Encoder<br>Text Decoder<br>**----- End of picture text -----**<br>


Figure 2. **Vid2Seq model overview** . We formulate dense event captioning as a sequence-to-sequence problem, using special _time tokens_ to allow the model to seamlessly understand and generate sequences of tokens containing both textual semantic information and temporal localization information grounding each text sentence in the video. In detail, all input video frames _x_ and the transcribed speech sequence _y_ are first processed with a Visual Encoder _f_ (a frozen Spatial Encoder _f[s]_ followed by a Temporal Encoder _f[t]_ ) and a Text Encoder _g_ (a Token Embedder _g[s]_ followed by a Transformer Encoder _g[t]_ ), respectively. Then the Text Decoder _h_ (composed of a Token Embedder _h[s]_ , a Transformer Encoder _h[t]_ and a Language Modeling Head _h[l]_ ) autoregressively generates the output event sequence _z_ by cross-attending to the visual and speech embeddings _x[t]_ and _y[t]_ . 

efits on multiple downstream datasets. Huang _et al_ . [35] explore pretraining on narrated instructional videos, but only consider event captioning using ground truth proposals as their model does not handle localization. Finally, [35, 133] explore pretraining on a domain specific text-only dataset [45]. In contrast, we propose to pretrain on a generic video corpus [121] and show benefits on various domains. **Unifying tasks as language modeling.** Recent works [11– 13, 15, 17, 44, 59, 101, 117, 132] have shown that it is possible to cast various computer vision problems as a language modeling task, addressing object detection [11], grounded image captioning [117] or visual grounding [132]. In this work we also cast visual localization as a language modeling task. However, unlike prior work focused on imagelevel spatial localization, we address the different problem of event localization _in time_ , in untrimmed videos. 

## **3. Method** 

The goal of dense video captioning is to temporally localize and describe with natural language _all_ events in an untrimmed input video. Therefore a key challenge is to effectively model the relationships between the different events in the video, as for example, it is easier to predict that the dogs are pulling the sled if we know that the man has just fastened a dog (see Figure 1 (right)). Furthermore, due to the dense nature of the task, there can be many events 

in a long video and the requirement is to output a natural language caption for each event. Hence, another key challenge is that the manual collection of annotations for this task is particularly expensive. To tackle these challenges, we first develop a unified multi-modal model that jointly predicts event boundaries and captions as a single sequence of tokens, as explained in Section 3.1 and Figure 2. Second, we design a pretraining strategy that effectively leverages cross-modal supervision in the form of transcribed speech from unlabeled narrated videos by reformulating sentence boundaries as pseudo event boundaries, as presented in Section 3.2 and Figure 3. 

## **3.1. Model** 

We wish to design a model for dense video captioning that can capture relationships between events using visual and (transcribed) speech cues in order to effectively localize and describe these events in untrimmed minutes-long videos. To tackle this challenge, we cast dense video captioning as a sequence-to-sequence problem where the input and output sequences contain both the semantic information about the event in the form of natural language descriptions and the temporal localization of the events in the form of temporal timestamps. In addition, to best leverage both the visual and the language signal, we develop an appropriate multi-modal encoder-decoder architecture. As illustrated 

3 

in Figure 2, our architecture takes as input video frames _x_ = _{xi}[F] i_ =1[together with the transcribed speech sequence] _y_ = _{yj}[S] j_ =1[.][The output of our model is an event sequence] _z_ = _{zk}[L] k_ =1[,][where][each][event][contains][both][its][textual] description and timestamps corresponding to the temporal event locations in the video. Below we explain the structure of the transcribed speech and event sequences constructed for our model as well as details of our model architecture. 

**Sequence construction.** To model inter-event relationships in dense event captioning annotations (or the readilyavailable transcribed narration, see Section 3.2), we cast dense video captioning as predicting a single output sequence of tokens _z_ . This output event sequence is constructed by leveraging a text tokenizer augmented with special _time tokens_ . Furthermore, we enable our architecture to jointly reason about the semantic and temporal information provided in the transcript of the input narration by constructing the input transcript sequence _y_ in a similar manner as the event sequence _z_ . Details are given next. _**Time tokenization.**_ We start from a text tokenizer with a vocabulary size _V_ , and augment it with _N_ additional time tokens, resulting in a tokenizer with _V_ + _N_ tokens. The time tokens represent relative timestamps in a video, as we quantize a video of duration _T_ into _N_ equally-spaced timestamps. In detail, we use the SentencePiece tokenizer [47] with vocabulary size _V_ = 32 _,_ 128 and _N_ = 100. _**Event sequence.**_ Our introduced tokenizer enables us to construct sequences that contain both video timestamps and text video descriptions. We next explain how we construct the output event sequence _z_ . Note that videos have a variable number of events in standard dense video captioning datasets [35, 46, 130]. Each event _k_ is characterized by a text segment, a start time and an end time. We first construct for each event _k_ a sequence by concatenating its start time token _tstartk_ , its end time token _tendk_ and its text tokens [ _zk_ 1 _, ..., zklk_ ]. Then we order all these sequences in increasing order of their start times and concatenate them. In practice, each text segment ends with a dot symbol indicating the separation between different events. Finally, the event sequence is obtained by prepending and appending a BOS and an EOS tokens to indicate the start and the end of sequence, respectively, _i.e_ . _z_ = [ _BOS, tstart_ 1 _, tend_ 1 _, z_ 11 _, ..., z_ 1 _l_ 1 _, tstart_ 2 _, ..., EOS_ ]. _**Transcribed speech sequence.**_ To enable the model to use both the transcribed speech and its corresponding timestamps, we convert the speech transcript into a speech sequence _y_ similarly as the input training dense event captions _z_ . This is done by segmenting the raw speech transcript into sentences with the Google Cloud API[1] , and using each transcribed speech sentence with its corresponding timestamps analogously as an event in the previously explained process. 

> 1https://cloud.google.com/speech-to-text/docs/automatic-punctuation. 

**Architecture.** We wish to design an architecture that can effectively model relationships between different events in untrimmed minutes-long videos. To tackle this challenge, we propose a multi-modal encoder-decoder architecture, illustrated in Figure 2, that gradually refines and outputs the event sequence described above. In detail, given an untrimmed minutes-long video, the visual encoder _f_ embeds its frames while the text encoder _g_ embeds transcribed speech and the corresponding timestamps. Then a text decoder _h_ predicts event boundaries and text captions using the visual and transcribed speech embeddings. The individual modules are described next. 

_**Visual encoder.**_ The visual encoder operates on a sequence of _F_ frames _x ∈_ R _[F][ ×][H][×][W][ ×][C]_ where _H_ , _W_ and _C_ are the height, width and the number of channels of each frame. A visual backbone _f[s]_ first encodes each frame separately and outputs frame embeddings _x[s]_ = _f[s]_ ( _x_ ) _∈_ R _[F][ ×][d]_ , where _d_ is the embedding dimension. Then a transformer encoder [94] _f[t]_ models temporal interactions between the different frames, and outputs _F_ contextualized visual embeddings _x[t]_ = _f[t]_ ( _x[s]_ + _x[p]_ ) _∈_ R _[F][ ×][d]_ , where _x[p] ∈_ R _[F][ ×][d]_ are learnt temporal positional embeddings, which communicate time information from visual inputs to the model. In detail, the visual backbone is CLIP ViT-L/14 [22, 77] at resolution 224 _×_ 224 pixels, pretrained to map images to text descriptions with a contrastive loss on Web-scraped imagetext pairs. We keep the backbone frozen for efficiency. 

_**Text encoder.**_ The text encoder operates on a transcribed speech sequence of _S_ tokens _y ∈{_ 1 _, ..., V_ + _N }[S]_ , where _V_ is the text vocabulary size, _N_ is the size of the vocabulary of time tokens and _S_ is the number of tokens in the transcribed speech sequence. Note that the transcribed speech sequence includes time tokens to input the temporal information from the transcribed speech into the model. An embedding layer _g[s] ∈_ R[(] _[V]_[ +] _[N]_[)] _[×][d]_ embeds each token independently and outputs semantic embeddings _y[s]_ = _g[s]_ ( _y_ ) _∈_ R _[S][×][d]_ . Then a transformer encoder _g[t]_ computes interactions in the transcribed speech sequence and outputs _S_ contextualized speech embeddings _y[t]_ = _g[t]_ ( _y[s]_ ) _∈_ R _[S][×][d]_ . 

_**Text decoder.**_ The text decoder generates the event sequence _z_ by using the encoder embeddings, which are obtained by concatenating the visual and speech embeddings _x[t]_ and _y[t]_ . The text decoder is based on a causal transformer decoder _h[t]_ that cross-attends to the encoder outputs, and at each autoregressive step _k_ , self-attends to the previously generated tokens ˆ _z<k[t]_[to output a contextualized representa-] tion _zk[t]_[=] _[ h][t]_[(] _[h][s]_[(ˆ] _[z] <k[t]_[)] _[, x][t][, y][t]_[)] _[ ∈]_[R] _[d]_[ where] _[ h][s][∈]_[R][(] _[V]_[ +] _[N]_[)] _[×][d]_ is the decoder token embedding layer. Then a language modeling head _h[l] ∈_ R _[d][×]_[(] _[V]_[ +] _[N]_[)] predicts a probability distribution over the joint vocabulary of text and time tokens in order to predict the next token in the event sequence, _i.e_ . _zk[l]_[=] _[ h][l]_[(] _[z] k[t]_[)] _[ ∈]_[R] _[V]_[ +] _[N]_[.] 

_**Text initialization.**_ We initialize the text encoder and the 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Vid2Seq_Large-Scale_Pretraining_of_a_Visual_Language_Model_for_Dense_Video_Captioning_images/cvpr_2023_Vid2Seq_Large-Scale_Pretraining_of_a_Visual_Language_Model_for_Dense_Video_Captioning.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Input video frames Generative Task Transcribed speech sequence<br><3><5> Hey guys today I am<br>Vid2Seq going to teach you how to…<br><93><96> First slope, congratz!<br>Corrupted transcribed speech  Denoising Task Recovered transcribed<br>sequence speech sequence<br><3><5> Hey guys today I am  Vid2Seq [X] teach you how to ski [Y]<br>going to [X] … <93> [Y] congratz! <96> First slope [Z]<br>**----- End of picture text -----**<br>


Figure 3. **Pretraining tasks** . To train Vid2Seq on unlabeled narrated videos, we design two pretraining objectives. **Top** : generative objective, given visual inputs _x_ only, the task is to generate the transcribed speech sequence _y_ . **Bottom** : denoising objective, given visual inputs _x_ and the corrupted speech sequence ˜ _y_ , the task ¯ is to generate the sequence of recovered speech segments _y_ . 

text decoder with T5-Base [78] which has been pretrained on Web text corpora with a denoising loss. Therefore their implementation and parameters also closely follow T5-Base, _e.g_ . they use relative positional embeddings and share their token embedding layer _g[s]_ = _h[s] ∈_ R[(] _[V]_[ +] _[N]_[)] _[×][d]_ . 

## **3.2. Training** 

In this Section, we describe how we leverage a large amount of unlabeled narrated videos to train the previously described dense event captioning model. We first present the pretraining method used to effectively train Vid2Seq using cross-modal supervision in readily-available narrated videos in Section 3.2.1 and Figure 3. Then we explain how we finetune our architecture for various downstream tasks including dense event captioning in Section 3.2.2. 

## **3.2.1 Pretraining on untrimmed narrated videos** 

We wish to leverage narrated videos for pretraining as they are easily available at scale [72,121]. However these videos do not contain dense event captioning annotations. Therefore we use as supervisory signal the transcribed speech sentences and their corresponding timestamps. As speech transcripts are not always visually grounded and often temporally misaligned [32, 43, 71], we note that they only provide _weak_ supervision. Furthermore, speech transcripts drastically differ from dense event captioning annotations. For instance, in the YT-Temporal-1B dataset [121], a video contains 120 speech sentences on average which is an order of magnitude more than the number of events in standard dense video captioning datasets [35, 46, 130]. Our Vid2Seq model is particularly suitable for using such weak supervision as it constructs the speech sequence similarly as a manually annotated event sequence, and jointly contextualizes the speech boundaries and semantic information on the level of potentially minutes-long videos (see Section 3.1) rather than at a shorter clip-level, enabling our model to learn long-term relationships between the different speech segments: in experiments we show that pretraining on entire minutes-long videos is highly beneficial. 

We next describe the two proposed training objectives, 

which are both based on a maximum likelihood objective. Formally, given visual inputs _x_ , encoder text sequence _y_ and a decoder target text sequence _z_ , both objectives are based on minimizing the following loss: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Vid2Seq_Large-Scale_Pretraining_of_a_Visual_Language_Model_for_Dense_Video_Captioning_images/cvpr_2023_Vid2Seq_Large-Scale_Pretraining_of_a_Visual_Language_Model_for_Dense_Video_Captioning.pdf-0005-09.png)


where _L_ is the length of the decoder target sequence, _wk_ is the weight for k-th token in the sequence, which we set to _wk_ = 1 _∀k_ in practice, _θ_ denotes the trainable parameters in the model and _pθ_ is the output probability distribution over the vocabulary of text and time tokens. 

**Generative objective.** This objective uses the transcribed speech as a (pseudo-)supervisory signal to teach the decoder to predict a sequence of events given visual inputs. Given video frames _x_ , which are fed to the encoder, the decoder has to predict the transcribed speech sequence _y_ (see Figure 3), which serves as a proxy dense event captioning annotation. Note that no text input is given to the encoder for this task as using transcribed speech both as input and target would lead the model to learn text-only shortcuts. 

**Denoising objective.** As no text input is given to the encoder for the generative proxy task, the generative objective only trains the visual encoder and the text decoder, but not the text encoder. However when our model is used for dense video captioning, the text encoder has a significant importance as it encodes speech transcripts. Hence we introduce a denoising objective that aims at jointly aligning the visual encoder, the text encoder and the text decoder. Inspired by T5 [78] in the text domain, we randomly mask spans of (text and time) tokens in the transcribed speech sequence with a probability _P_ and an average span length _M_ . The encoder input is composed of the video frames _x_ together with the corrupted speech sequence _y_ ˜, which contains sentinel tokens that uniquely identify the masked spans. The decoder then has to predict a sequence _y_ ¯ constructed with the corresponding masked spans for each sentinel token, based on visual inputs _x_ and speech context _y_ ˜ (see Figure 3). 

## **3.2.2 Downstream task adaptation** 

Our architecture and task formulation enables us to tackle dense video captioning with a generic language modeling training objective and inference procedure. Note that as a by-product of our generic architecture, our model can also be used to generate paragraphs about entire videos by simply removing the time tokens from the output sequence, and can also be easily adapted to video clip captioning with the same finetuning and inference recipe. 

**Finetuning.** To finetune our model for dense video captioning, we use a maximum likelihood objective based on the event sequence (see Equation 1). Given video frames _x_ and speech transcripts _y_ , the decoder has to predict the event sequence _z_ . 

5 

**Inference.** The text decoder autoregressively generates the event sequence by sampling from the model likelihood. In practice, we use beam search as we find that it improves the captioning quality compared with argmax sampling or nucleus sampling. Finally, the event sequence is converted into a set of event predictions by simply reversing the sequence construction process. 

## **4. Experiments** 

This section demonstrates the effectiveness of our pretrained Vid2Seq model and compares our method to the state of the art. We first outline our experimental setup in Section 4.1. We then present ablation studies in Section 4.2. The comparison to the state of the art in dense video captioning, video paragraph captioning and video clip captioning is presented in Section 4.3. Next, we present results in a new few-shot dense video captioning setting in Section 4.4. Finally, we show qualitative results in Section 4.5. 

## **4.1. Experimental setup** 

**Datasets.** For pretraining, following prior work showing the benefits of pretraining on a diverse and large dataset [122], we use the **YT-Temporal-1B** dataset [121], which includes 18 million narrated videos collected from YouTube. We evaluate Vid2Seq on three downstream dense video captioning datasets: YouCook2 [130], ViTT [35] and ActivityNet Captions [46]. **YouCook2** has 2K untrimmed videos of cooking procedures. On average, each video lasts 320s and is annotated with 7.7 temporally-localized sentences. **ViTT** consists of 8K untrimmed instructional videos. On average, each video lasts 250s and is annotated with 7.1 temporally-localized short tags. **ActivityNet Captions** contains 20k untrimmed videos of various human activities. On average, each video lasts 120s and is annotated with 3.7 temporally-localized sentences. For video clip captioning, we use two standard benchmarks, **MSR-VTT** [109] and **MSVD** [9]. For all datasets, we follow the standard splits for training, validation and testing. Note that we only use videos available on YouTube at the time of the work, resulting in 10 to 20% less videos than in the original datasets. **Implementation details.** We extract video frames at 1FPS, and subsample or pad the sequence of frames to _F_ frames where we set _F_ = 100. The text encoder and decoder sequence are truncated or padded to _L_ = _S_ = 1000 tokens. Our model has 314M trainable parameters. We use the Adam optimizer [42]. We pretrain our model for 200,000 iterations with a batch size of 512 videos split on 64 TPU v4 chips, which lasts a day. We sum both pretraining objectives with equal weighting to get our final pretraining loss. More details are included in Appendix Section B. 

**Evaluation metrics.** For captioning, we use CIDEr [95] (C) and METEOR [6] (M). For dense video captioning, we follow the commonly used evaluation tool [46] which calculates matched pairs between generated events and the 

|Pretraining input<br>Untrimmed<br>Time tokens|YouCook2<br>S<br>C<br>F1|ActivityNet<br>S<br>C<br>F1|
|---|---|---|
|1.<br>_No pretraining_<br>2.<br>✗<br>✗<br>3.<br>✓<br>✗<br>4.<br>✓<br>✓|4.0<br>18.0<br>18.1<br>5.5<br>27.8<br>20.5<br>6.7<br>35.0<br>23.3<br>**7.9**<br>**47.1**<br>**27.3**|5.4<br>18.8<br>49.2<br>5.5<br>26.5<br>52.1<br>5.6<br>27.4<br>52.2<br>**5.8**<br>**30.1**<br>**52.4**|



Table 1. **Ablation showing the impact of using untrimmed videos and adding time tokens during pretraining.** When we use untrimmed video-speech inputs, time information from transcribed speech sentence boundaries is integrated via time tokens. 

ground truth across IoU thresholds of _{_ 0.3, 0.5, 0.7, 0.9 _}_ , and compute captioning metrics over the matched pairs. However, these metrics do not take into account the story of the video. Therefore we also use SODA ~~c~~ [27] (S) for an overall dense video captioning evaluation. To further isolate the evaluation of event localization, we report the average precision and average recall across IoU thresholds of _{_ 0.3, 0.5, 0.7, 0.9 _}_ and their harmonic mean, the F1 Score. 

## **4.2. Ablation studies** 

The default Vid2Seq model predicts both text and time tokens, uses both visual frames and transcribed speech as input, builds on the T5-Base language model, and is pretrained on untrimmed videos from YT-Temporal-1B with both the generative and denoising losses. Below we ablate the importance of each of these factors on the downstream dense video captioning performance by reporting results on YouCook2 and ActivityNet Captions validation sets. 

**Pretraining on untrimmed narrated videos by exploiting transcribed speech sentence boundaries.** In Table 1, we evaluate the effectiveness of our pretraining task formulation that uses untrimmed videos and integrates sentence boundaries of transcribed speech via time tokens. In contrast, most video clip captioning pretraining methods [35, 70, 80] use short, trimmed, video-speech segments for pretraining. We adapt this strategy in our model and find that it indeed yields significant performance improvements over the baseline that uses no video-text pretraining (row 2 vs row 1). However, larger improvements are obtained by using untrimmed video-speech inputs (row 3 vs row 2). Moreover, using time tokens to integrate time information from transcribed speech drastically improves performance (row 4 vs row 3). This shows the benefits of exploiting sentence boundaries of transcribed speech via time tokens and of using untrimmed videos during pretraining. In Appendix Section C.2, we show additional ablations to quantify how the performance improves by pretraining on longer narrated videos that contain more speech sentences. 

**Input modalities and pretraining objectives.** In Table 2, we analyze the importance of input modalities and pretraining tasks on the downstream dense video captioning performance. The model with visual inputs only (no transcribed speech as input) benefits significantly from pretraining with the generative objective (row 3 vs row 1). This shows the 

6 

|Finetuning Input<br>Visual<br>Speech|Pretraining loss<br>Generative Denoising|YouCook2<br>S<br>C<br>F1|ActivityNet<br>S<br>C<br>F1|
|---|---|---|---|
|1.<br>✓<br>✗<br>_No pretraining_<br>3.0 15.6 15.4<br>5.4 14.2 46.5<br>2.<br>✓<br>✓<br>_No pretraining_<br>4.0 18.0 18.1<br>5.4 18.8 49.2<br>3.<br>✓<br>✗<br>✓<br>✗<br>5.7 25.3 23.5<br>**5.9 30.2** 51.8<br>4.<br>✓<br>✓<br>✓<br>✗<br>2.5 10.3 15.9<br>4.8 17.0 48.8<br>5.<br>✓<br>✓<br>✓<br>✓<br>**7.9 47.1 27.3**<br>5.8 30.1 **52.4**||||



Table 2. **Effect of input modalities and pretraining losses.** 

|Table 2. **Effect of inpu**|**t modalities and pr**|**etraining losses.**|
|---|---|---|
|Captioning Pretraining|YouCook2<br>Recall Precision<br>F1|ActivityNet<br>Recall Precision<br>F1|
|1.<br>✗<br>✗<br>2.<br>✓<br>✗<br>3.<br>✗<br>✓<br>4.<br>✓<br>✓|17.8<br>19.4<br>17.7<br>17.2<br>20.6<br>18.1<br>25.7<br>21.4<br>22.8<br>**27.9**<br>**27.8**<br>**27.3**|47.3<br>57.9<br>52.0<br>42.5<br>**64.1**<br>49.2<br>52.5<br>53.0<br>51.1<br>**52.7**<br>53.9<br>**52.4**|



Table 3. **Effect of joint captioning and localization on the localization performance.** The variant that does not caption corresponds to a localization-only variant that only predicts time tokens. 

|Language<br>Model|Pretraining<br># Videos Dataset|YouCook2<br>S<br>C<br>F1|ActivityNet<br>S<br>C<br>F1|
|---|---|---|---|
|1.<br>T5-Small<br>2.<br>T5-Base<br>3.<br>T5-Base<br>4.<br>T5-Base<br>5.<br>T5-Base<br>6.<br>T5-Base<br>7.<br>T5-Base|15M<br>YTT<br>_∅_<br>_∅_<br>15K<br>YTT<br>150K<br>YTT<br>1M5<br>YTT<br>1M<br>HTM<br>15M<br>YTT|6.1 31.1 24.3<br>4.0 18.0 18.1<br>6.3 35.0 24.4<br>7.3 40.1 26.7<br>7.8 45.5 26.8<br>**8.3 48.3**<br>26.6<br>7.9 47.1<br>**27.3**|5.5<br>26.5<br>52.2<br>5.4<br>18.8<br>49.2<br>5.1<br>24.4<br>49.9<br>5.4<br>27.2<br>51.3<br>5.6<br>28.7<br>52.2<br>**5.8**<br>28.8<br>**53.1**<br>**5.8**<br>**30.1**<br>52.4|



Table 4. **Effect of language model size and pretraining data.** HTM: HowTo100M [72], YTT: YT-Temporal-1B [121]. 

effectiveness of using the transcribed speech as a proxy annotation for dense video captioning pretraining. However, this model is pretrained with visual inputs only and its performance largely drops when it is finetuned with both visual and transcribed speech inputs (row 4 vs row 3). With both modalities, adding the denoising loss strongly benefits our model (row 5 vs rows 4 and 2). We conclude that the denoising objective benefits multi-modal reasoning. 

**Effect of captioning on localization.** In Table 3, we compare the event localization performance of our model with a localization-only variant that only predicts event boundaries. We find that the model that jointly predicts event boundaries and captions localizes better and benefits more from pretraining than the localization-only baseline (row 4 vs row 3), which demonstrates the importance of contextualizing the noisy timestamps of the transcribed speech with the speech semantic content during pretraining. 

**Model size and pretraining data.** In Table 4, we show that the language model size has a great importance on the performance, as the model with T5-Base outperforms its variant with T5-Small (row 7 vs row 1). We also evaluate the importance of the size of the pretraining dataset of narrated videos by constructing subsets such that larger subsets include the smaller ones. We find that scaling up the size of the pretraining dataset is beneficial, and that our pretraining method yields important benefits when only using 150K narrated videos for pretraining (row 4). We 

|Method|Backbone|YouCook2 (val)<br>S<br>C<br>M|ViTT (test)<br>S<br>C<br>M|ActivityNet (val)<br>S<br>C<br>M|
|---|---|---|---|---|
|MT [131]<br>ECHR [103]<br>PDVC [102]<br>PDVC [102]_†_<br>UEDVC [125]<br>E2ESG [133]<br>Vid2Seq (Ours)|TSN<br>C3D<br>TSN<br>CLIP<br>TSN<br>C3D<br>CLIP|—<br>6.1<br>3.2<br>—<br>—<br>3.8<br>4.4<br>22.7<br>4.7<br>4.9<br>28.9<br>5.7<br>—<br>—<br>—<br>—<br>25.0*<br>3.5<br>**7.9**<br>**47.1**<br>**9.3**|—<br>—<br>—<br>—<br>—<br>—<br>—<br>—<br>—<br>—<br>—<br>—<br>—<br>—<br>—<br>—<br>25.0<br>8.1<br>**13.5**<br>**43.5**<br>**8.5**|—<br>9.3<br>5.0<br>3.2<br>14.7<br>7.2<br>5.4<br>29.0<br>8.0<br>**6.0**<br>29.3<br>7.6<br>5.5<br>—<br>—<br>—<br>—<br>—-<br>5.8<br>**30.1**<br>**8.5**|



Table 5. Comparison to the state of the art for dense video captioning. * Results provided by the authors. _[†]_ Results of our experiments using the official codebase. 

|Method|Backbone|YouCook2 (val)<br>Recall<br>Precision|ViTT (test)<br>Recall<br>Precision|ActivityNet (val)<br>Recall<br>Precision|
|---|---|---|---|---|
|PDVC [102]<br>PDVC [102]_†_<br>UEDVC [125]<br>E2ESG [133]<br>Vid2Seq (Ours)|TSN<br>CLIP<br>TSN<br>C3D<br>CLIP|—<br>—<br>—<br>—<br>—<br>—<br>20.7*<br>20.6*<br>**27.9**<br>**27.8**|—<br>—<br>—<br>—<br>—<br>—<br>32.2*<br>32.1*<br>**42.6**<br>**46.2**|55.4<br>58.1<br>53.2<br>54.7<br>**59.0**<br>**60.3**<br>—<br>—<br>52.7<br>53.9|



Table 6. Comparison to the state of the art for event localization. * Results provided by the authors. _†_ Results of our experiments using the official codebase. 

further show that our pretraining method generalizes well to the HowTo100M dataset [72]. The model pretrained on HowTo100M (row 6) actually achieves best results on YouCook2, as these datasets are from a similar domain. Finally, we ablate the importance of the size and pretraining of the visual backbone in Appendix Section C.2. 

## **4.3. Comparison to the state of the art** 

**Dense video captioning.** In Table 5, we compare our approach to state-of-the-art dense video captioning methods using cross-entropy training[1] on the YouCook2, ViTT and ActivityNet Captions datasets. Vid2Seq sets new state of the art on all three datasets. In particular, our method improves the CIDEr metric by 18.2 and 0.8 points on YouCook2 and ActivityNet Captions over PDVC. Our method also outperforms E2ESG [133] which uses indomain text-only pretraining on Wikihow. These results demonstrate the strong dense event captioning ability of our pretrained Vid2Seq model. 

**Event localization.** In Table 6, we evaluate the event localization performance of our dense video captioning model in comparison with prior work. On both YouCook2 and ViTT, Vid2Seq outperforms prior work [133] tackling dense video captioning as a single sequence generation task. However, our model underperforms compared to PDVC [102] and UEDVC [102] on ActivityNet Captions. We emphasize that our approach integrates less prior knowledge about temporal localization than both these approaches, which include task specific components such as event counters [102] or separately train a model for the localization subtask [125]. **Video paragraph captioning.** In Table 7, we compare our approach to state-of-the-art video paragraph captioning methods on the YouCook2 and ActivityNet Cap- 

> 1We do not include methods directly optimizing the test metric [20,73]. 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Vid2Seq_Large-Scale_Pretraining_of_a_Visual_Language_Model_for_Dense_Video_Captioning_images/cvpr_2023_Vid2Seq_Large-Scale_Pretraining_of_a_Visual_Language_Model_for_Dense_Video_Captioning.pdf-0008-00.png)


**----- Start of picture text -----**<br>
Input  Next Oh is Christina Oh Beck full most consistent off the top women javelin throwers Well, that's Christina Oh beg for  She's got over the years know what<br>Speech around at the moment. another … what a wonderful record. major gold medals until now.<br>very fine.<br>Input<br>Frames<br>GT An athlete is seen standing ready before a large track. The woman throws a javelin off into the distance and is shown again<br>afterwards. She throws her hands up to<br>cheer and wraps herself in a<br>flag.<br>Vis2Seq A woman runs with a javelin.  She throws it onto the field. She throws a second javelin. She waves to<br>the crowd and<br>holds up a flag.<br>**----- End of picture text -----**<br>


Figure 4. Example of dense event captioning predictions of Vid2Seq on ActivityNet Captions validation set, compared with ground-truth. 

|Method|Backbone|YouCook2 (val)<br>C<br>M|YouCook2 (val)<br>C<br>M|ActivityNet (val-ae)<br>C<br>M|ActivityNet (val-ae)<br>C<br>M|
|---|---|---|---|---|---|
|_With GT Proposals_||||||
|VTransformer [131]<br>Transformer-XL [18]<br>MART [50]|V (ResNet-200) + F<br>V (ResNet-200) + F<br>V (ResNet-200) + F|32.3<br>26.4<br>35.7|15.7<br>14.8<br>15.9|22.2<br>21.7<br>23.4|15.6<br>15.1<br>15.7|
|GVDSup [128]<br>AdvInf [76]<br>PDVC[102]|V (ResNet-101) + F + O<br>V (ResNet-101) + F + O<br>V + F(TSN)|—<br>—<br>—|—<br>—<br>—|22.9<br>21.0<br>27.3|16.4<br>16.6<br>15.9|
|_With Learnt Proposals_||||||
|MFT [107]<br>PDVC [102]|V + F (TSN)<br>V + F (TSN)|—<br>—|—<br>—|19.1<br>20.5|14.7<br>15.8|
|PDVC [102]_†_|V (CLIP)|—|—|23.6|15.9|
|Vid2Seq (Ours)|V (CLIP)|**50.1**|**24.0**|**28.0**|**17.0**|



Table 7. Comparison to the SoTA for video paragraph captioning. _†_ Results of our experiments using the official codebase. V/F/O refers to visual/flow/object features. 

|Method|Trained<br>Parameters<br>Pretraining<br>Data|MSR-VTT (test)<br>C<br>M|MSVD (test)<br>C<br>M|
|---|---|---|---|
|ORG-TRL [126]<br>SwinBERT [63]<br>Vid2Seq (Ours)|—<br>_∅_<br>229M<br>_∅_<br>314M<br>_∅_|50.9<br>28.8<br>53.8<br>29.9<br>57.2<br>30.0|95.2<br>36.4<br>120.6 41.3<br>120.3 41.4|
|MV-GPT [80]<br>Vid2Seq (Ours)|354M<br>HowTo100M<br>314M<br>HowTo100M|60.0<br>29.9_∗_<br>61.5<br>30.4|—<br>—<br>140.6 44.5|
|Vid2Seq (Ours)|314M<br>YT-Temporal-1B|**64.6**<br>**30.8**|**146.2 45.3**|



Table 8. Comparison to the SoTA for video clip captioning. * indicates results re-evaluated by the same evaluation toolkit. 

|Data|YouCook2<br>S<br>C<br>M|ViTT<br>S<br>C<br>M|ActivityNet<br>S<br>C<br>M|
|---|---|---|---|
|1.<br>1%<br>2.<br>10%<br>3.<br>50%<br>4.<br>100%|2.4<br>10.1<br>3.3<br>3.8<br>18.4<br>5.2<br>6.2<br>32.1<br>7.6<br>**7.9**<br>**47.1**<br>**9.3**|2.0<br>7.4<br>1.9<br>10.7<br>28.6<br>6.0<br>12.5<br>38.8<br>7.8<br>**13.5**<br>**43.5**<br>**8.5**|2.2<br>6.2<br>3.2<br>4.3<br>20.0<br>6.1<br>5.4<br>27.5<br>7.8<br>**5.8**<br>**30.1**<br>**8.5**|



Table 9. **Few-shot dense event captioning** , by finetuning Vid2Seq using a small fraction of the downstream training dataset. 

tions datasets. Vid2Seq outperforms all prior methods on both datasets, including the ones using ground-truth event boundary proposals at inference time [18, 50, 76, 102, 128, 131], showing strong video paragraph captioning ability. **Video clip captioning.** In Table 8, we compare our approach to state-of-the-art video clip captioning methods on the MSR-VTT and MSVD datasets. Vid2Seq improves over prior methods in their respective pretraining data setting while using a comparable number of trained parameters. We conclude that our pretrained Vid2Seq model generalizes well to the standard video clip captioning setting. 

## **4.4. Few-shot dense video captioning** 

To further evaluate the generalization capabilities of our pretrained Vid2Seq model, we propose a new few-shot dense video captioning setting where we finetune Vid2Seq using only a fraction of the downstream training dataset. From Table 9, we observe important improvements when using 10% compared to 1% of training data (row 2 vs 1). In Appendix Section C.1 we further show that pretraining is essential in this few-shot setting. 

## **4.5. Qualitative examples** 

In Figure 4, we show an example of dense event captioning predictions from Vid2Seq. This example shows that our model can predict meaningful event boundaries and captions, and that the predicted captions and boundaries differ considerably from the transcribed speech input (showing the importance of the visual tokens in the input). More examples are provided in Appendix Section A. 

## **5. Conclusion** 

We introduced Vid2Seq, a visual language model that performs dense video captioning by generating a single sequence of tokens including both text and time tokens given multi-modal inputs. We showed that Vid2Seq benefits from large-scale pretraining on unlabeled untrimmed narrated videos by leveraging transcribed speech sentences and corresponding temporal boundaries. Vid2Seq achieves stateof-the-art results on various dense event captioning datasets, as well as multiple video paragraph captioning and standard video clip captioning benchmarks. Finally, we believe the sequence-to-sequence design of Vid2Seq has the potential to be extended to a wide range of _other_ video tasks such as temporally-grounded video question answering [51, 56, 57] or temporal action localization [16, 67, 123]. 

**Acknowledgements.** The work was partially funded by a Google gift, the French government under management of Agence Nationale de la Recherche as part of the ”Investissements d’avenir” program, reference ANR-19-P3IA-0001 (PRAIRIE 3IA Institute), the Louis Vuitton ENS Chair on Artificial Intelligence, the European Regional Development Fund under project IMPACT (reg. no. CZ.02.1.01/0.0/0.0/15 003/0000468). We thank Anurag Arnab, Minsu Cho, Anja Hauth, Ashish Thapliyal, Bo Pang, Bryan Seybold and the entire Ganesha team for helpful discussions. 

8 

## **References** 

- [1] Vid2Seq project webpage. https://antoyang. github.io/vid2seq.html. 1, 2 

- [2] Sami Abu-El-Haija, Nisarg Kothari, Joonseok Lee, Paul Natsev, George Toderici, Balakrishnan Varadarajan, and Sudheendra Vijayanarasimhan. Youtube-8m: A largescale video classification benchmark. _arXiv preprint arXiv:1609.08675_ , 2016. 14 

- [3] Hassan Akbari, Liangzhe Yuan, Rui Qian, Wei-Hong Chuang, Shih-Fu Chang, Yin Cui, and Boqing Gong. Vatt: Transformers for multimodal self-supervised learning from raw video, audio and text. _NeurIPS_ , 2021. 2 

- [4] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In _NeurIPS_ , 2022. 1, 2 

- [5] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In _ICCV_ , 2021. 2 

- [6] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In _Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization_ , 2005. 6 

- [7] Meng Cao, Tianyu Yang, Junwu Weng, Can Zhang, Jue Wang, and Yuexian Zou. Locvtp: Video-text pre-training for temporal localization. In _ECCV_ , 2022. 2 

- [8] Aman Chadha, Gurneet Arora, and Navpreet Kaloty. iPerceive: Applying common-sense reasoning to multi-modal dense video captioning and video question answering. In _WACV_ , 2021. 2 

- [9] David L Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In _ACL_ , 2011. 2, 6, 14 

- [10] Shaoxiang Chen and Yu-Gang Jiang. Towards bridging event captioner and sentence localizer for weakly supervised dense event captioning. In _CVPR_ , 2021. 2 

- [11] Ting Chen, Saurabh Saxena, Lala Li, David J Fleet, and Geoffrey Hinton. Pix2seq: A language modeling framework for object detection. In _ICLR_ , 2022. 1, 3 

- [12] Ting Chen, Saurabh Saxena, Lala Li, Tsung-Yi Lin, David J Fleet, and Geoffrey Hinton. A unified sequence interface for vision tasks. In _NeurIPS_ , 2022. 3 

- [13] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. PaLI: A jointly-scaled multilingual language-image model. _arXiv preprint arXiv:2209.06794_ , 2022. 1, 3 

- [14] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. UNITER: Universal image-text representation learning. In _ECCV_ , 2020. 2 

- [15] Zhiyang Chen, Yousong Zhu, Zhaowen Li, Fan Yang, Wei Li, Haixin Wang, Chaoyang Zhao, Liwei Wu, Rui Zhao, Jinqiao Wang, et al. Obj2seq: Formatting objects as se- 

quences with class prompt for visual tasks. In _NeurIPS_ , 2022. 3 

- [16] Feng Cheng and Gedas Bertasius. TALLformer: Temporal action localization with long-memory transformer. In _ECCV_ , 2022. 8 

- [17] Jaemin Cho, Jie Lei, Hao Tan, and Mohit Bansal. Unifying vision-and-language tasks via text generation. In _ICML_ , 2021. 3 

- [18] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. Transformer-XL: Attentive language models beyond a fixed-length context. In _ACL_ , 2019. 8 

- [19] Mostafa Dehghani, Alexey Gritsenko, Anurag Arnab, Matthias Minderer, and Yi Tay. Scenic: A jax library for computer vision research and beyond. In _CVPR_ , 2022. 2 

- [20] Chaorui Deng, Shizhe Chen, Da Chen, Yuan He, and Qi Wu. Sketch, ground, and refine: Top-down dense video captioning. In _CVPR_ , 2021. 1, 2, 7 

- [21] Karan Desai, Gaurav Kaul, Zubin Aysola, and Justin Johnson. RedCaps: Web-curated image-text data created by the people, for the people. In _NeurIPS Datasets and Benchmarks_ , 2021. 2 

- [22] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In _ICLR_ , 2021. 4 

- [23] Zi-Yi Dou, Aishwarya Kamath, Zhe Gan, Pengchuan Zhang, Jianfeng Wang, Linjie Li, Zicheng Liu, Ce Liu, Yann LeCun, Nanyun Peng, et al. Coarse-to-fine visionlanguage pre-training with fusion in the backbone. In _NeurIPS_ , 2022. 2 

- [24] Zi-Yi Dou, Yichong Xu, Zhe Gan, Jianfeng Wang, Shuohang Wang, Lijuan Wang, Chenguang Zhu, Pengchuan Zhang, Lu Yuan, Nanyun Peng, et al. An empirical study of training end-to-end vision-and-language transformers. In _CVPR_ , 2022. 2 

- [25] Victor Escorcia, Fabian Caba Heilbron, Juan Carlos Niebles, and Bernard Ghanem. Daps: Deep action proposals for action understanding. In _ECCV_ , 2016. 2 

- [26] Tsu-Jui Fu, Linjie Li, Zhe Gan, Kevin Lin, William Yang Wang, Lijuan Wang, and Zicheng Liu. VIOLET: End-toend video-language transformers with masked visual-token modeling. _arXiv preprint arXiv:2111.12681_ , 2021. 2 

- [27] Soichiro Fujita, Tsutomu Hirao, Hidetaka Kamigaito, Manabu Okumura, and Masaaki Nagata. SODA: Story oriented dense video captioning evaluation framework. In _ECCV_ , 2020. 6 

- [28] Zhe Gan, Yen-Chun Chen, Linjie Li, Chen Zhu, Yu Cheng, and Jingjing Liu. Large-scale adversarial training for vision-and-language representation learning. In _NeurIPS_ , 2020. 2 

- [29] Jiyang Gao, Zhenheng Yang, Kan Chen, Chen Sun, and Ram Nevatia. Turn tap: Temporal unit regression network for temporal action proposals. In _ICCV_ , 2017. 2 

9 

- [30] Lianli Gao, Zhao Guo, Hanwang Zhang, Xing Xu, and Heng Tao Shen. Video captioning with attention-based lstm and semantic consistency. _IEEE Transactions on Multimedia_ , 2017. 2 

- [31] Yuying Ge, Yixiao Ge, Xihui Liu, Dian Li, Ying Shan, Xiaohu Qie, and Ping Luo. Bridging video-text retrieval with multiple choice questions. In _CVPR_ , 2022. 2 

- [32] Tengda Han, Weidi Xie, and Andrew Zisserman. Temporal alignment networks for long-term video. In _CVPR_ , 2022. 2, 5 

- [33] Fabian Caba Heilbron, Juan Carlos Niebles, and Bernard Ghanem. Fast temporal activity proposals for efficient detection of human actions in untrimmed videos. In _CVPR_ , 2016. 2 

- [34] Xiaowei Hu, Zhe Gan, Jianfeng Wang, Zhengyuan Yang, Zicheng Liu, Yumao Lu, and Lijuan Wang. Scaling up vision-language pre-training for image captioning. In _CVPR_ , 2022. 2 

- [35] Gabriel Huang, Bo Pang, Zhenhai Zhu, Clara Rivera, and Radu Soricut. Multimodal pretraining for dense video captioning. In _AACL-IJCNLP_ , 2020. 1, 2, 3, 4, 5, 6, 14, 17 

- [36] Zhicheng Huang, Zhaoyang Zeng, Yupan Huang, Bei Liu, Dongmei Fu, and Jianlong Fu. Seeing out of the box: End-to-end pre-training for vision-language representation learning. In _CVPR_ , 2021. 2 

- [37] Vladimir Iashin and Esa Rahtu. A better use of audio-visual cues: Dense video captioning with bi-modal transformer. In _BMVC_ , 2020. 1, 2 

- [38] Vladimir Iashin and Esa Rahtu. Multi-modal dense video captioning. In _CVPR Workshops_ , 2020. 2 

- [39] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In _ICML_ , 2021. 2 

- [40] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. MDETR - modulated detection for end-to-end multi-modal understanding. In _ICCV_ , 2021. 2 

- [41] Wonjae Kim, Bokyung Son, and Ildoo Kim. ViLT: Visionand-language transformer without convolution or region supervision. In _ICML_ , 2021. 2 

- [42] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _ICLR_ , 2015. 6, 14 

- [43] Dohwan Ko, Joonmyung Choi, Juyeon Ko, Shinyeong Noh, Kyoung-Woon On, Eun-Sol Kim, and Hyunwoo J Kim. Video-text representation learning via differentiable weak temporal alignment. In _CVPR_ , 2022. 2, 5 

- [44] Alexander Kolesnikov, Andr´e Susano Pinto, Lucas Beyer, Xiaohua Zhai, Jeremiah Harmsen, and Neil Houlsby. Uvim: A unified modeling approach for vision with learned guiding codes. In _NeurIPS_ , 2022. 3 

- [45] Mahnaz Koupaee and William Yang Wang. Wikihow: A large scale text summarization dataset. _arXiv preprint arXiv:1810.09305_ , 2018. 3 

- [46] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In _ICCV_ , 2017. 1, 2, 4, 5, 6, 14 

- [47] Taku Kudo and John Richardson. Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In _ACL_ , 2018. 4 

- [48] Jie Lei, Tamara L Berg, and Mohit Bansal. Detecting moments and highlights in videos via natural language queries. In _NeurIPS_ , 2021. 2 

- [49] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L Berg, Mohit Bansal, and Jingjing Liu. Less is more: ClipBERT for video-and-language learning via sparse sampling. In _CVPR_ , 2021. 2 

- [50] Jie Lei, Liwei Wang, Yelong Shen, Dong Yu, Tamara L Berg, and Mohit Bansal. MART: Memory-augmented recurrent transformer for coherent video paragraph captioning. In _ACL_ , 2020. 2, 8, 14 

- [51] Jie Lei, Licheng Yu, Mohit Bansal, and Tamara L Berg. TVQA: Localized, compositional video question answering. In _EMNLP_ , 2018. 8 

- [52] Dongxu Li, Junnan Li, Hongdong Li, Juan Carlos Niebles, and Steven CH Hoi. Align and prompt: Video-andlanguage pre-training with entity prompts. In _CVPR_ , 2022. 2 

- [53] Gen Li, Nan Duan, Yuejian Fang, Ming Gong, Daxin Jiang, and Ming Zhou. Unicoder-VL: A universal encoder for vision and language by cross-modal pre-training. In _AAAI_ , 2020. 2 

- [54] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In _ICML_ , 2022. 2 

- [55] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. In _NeurIPS_ , 2021. 2 

- [56] Linjie Li, Yen-Chun Chen, Yu Cheng, Zhe Gan, Licheng Yu, and Jingjing Liu. HERO: Hierarchical encoder for video+language omni-representation pre-training. In _EMNLP_ , 2020. 2, 8 

- [57] Linjie Li, Jie Lei, Zhe Gan, Licheng Yu, Yen-Chun Chen, Rohit Pillai, Yu Cheng, Luowei Zhou, Xin Eric Wang, William Yang Wang, et al. VALUE: A multi-task benchmark for video-and-language understanding evaluation. In _NeurIPS Track on Datasets and Benchmarks_ , 2021. 8 

- [58] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In _CVPR_ , 2022. 2 

- [59] Wanhua Li, Zhexuan Cao, Jianjiang Feng, Jie Zhou, and Jiwen Lu. Label2Label: A language modeling framework for multi-attribute learning. In _ECCV_ , 2022. 3 

- [60] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pretraining for vision-language tasks. In _ECCV_ , 2020. 2 

- [61] Yehao Li, Ting Yao, Yingwei Pan, Hongyang Chao, and Tao Mei. Jointly localizing and describing events for dense video captioning. In _CVPR_ , 2018. 2 

10 

- [62] Chuming Lin, Jian Li, Yabiao Wang, Ying Tai, Donghao Luo, Zhipeng Cui, Chengjie Wang, Jilin Li, Feiyue Huang, and Rongrong Ji. Fast learning of temporal action proposal via dense boundary generator. In _AAAI_ , 2020. 2 

- [63] Kevin Lin, Linjie Li, Chung-Ching Lin, Faisal Ahmed, Zhe Gan, Zicheng Liu, Yumao Lu, and Lijuan Wang. SwinBERT: End-to-end transformers with sparse attention for video captioning. In _CVPR_ , 2022. 1, 2, 8 

- [64] Kevin Qinghong Lin, Alex Jinpeng Wang, Mattia Soldan, Michael Wray, Rui Yan, Eric Zhongcong Xu, Difei Gao, Rongcheng Tu, Wenzhe Zhao, Weijie Kong, et al. Egocentric video-language pretraining. In _NeurIPS_ , 2022. 2 

- [65] Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, and Shilei Wen. BMN: Boundary-matching network for temporal action proposal generation. In _ICCV_ , 2019. 2 

- [66] Tianwei Lin, Xu Zhao, Haisheng Su, Chongjing Wang, and Ming Yang. BSN: Boundary sensitive network for temporal action proposal generation. In _ECCV_ , 2018. 2 

- [67] Xiaolong Liu, Qimeng Wang, Yao Hu, Xu Tang, Shiwei Zhang, Song Bai, and Xiang Bai. End-to-end temporal action detection with transformer. In _IEEE Transactions on Image Processing_ , 2022. 8 

- [68] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. ViLBERT: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In _NeurIPS_ , 2019. 2 

- [69] Jiasen Lu, Vedanuj Goswami, Marcus Rohrbach, Devi Parikh, and Stefan Lee. 12-in-1: Multi-task vision and language representation learning. In _CVPR_ , 2020. 2 

- [70] Huaishao Luo, Lei Ji, Botian Shi, Haoyang Huang, Nan Duan, Tianrui Li, Xilin Chen, and Ming Zhou. UniViLM: A unified video and language pre-training model for multimodal understanding and generation. _arXiv preprint arXiv:2002.06353_ , 2020. 1, 6, 17 

- [71] Antoine Miech, Jean-Baptiste Alayrac, Lucas Smaira, Ivan Laptev, Josef Sivic, and Andrew Zisserman. End-to-end learning of visual representations from uncurated instructional videos. In _CVPR_ , 2020. 2, 5 

- [72] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. HowTo100M: Learning a text-video embedding by watching hundred million narrated video clips. In _ICCV_ , 2019. 2, 5, 7, 14 

- [73] Jonghwan Mun, Linjie Yang, Zhou Ren, Ning Xu, and Bohyung Han. Streamlined dense video captioning. In _CVPR_ , 2019. 2, 7 

- [74] Arsha Nagrani, Paul Hongsuck Seo, Bryan Seybold, Anja Hauth, Santiago Manen, Chen Sun, and Cordelia Schmid. Learning audio-video modalities from image captions. In _ECCV_ , 2022. 2 

- [75] Yingwei Pan, Ting Yao, Houqiang Li, and Tao Mei. Video captioning with transferred semantic attributes. In _CVPR_ , 2017. 2 

- [76] Jae Sung Park, Marcus Rohrbach, Trevor Darrell, and Anna Rohrbach. Adversarial inference for multi-sentence video description. In _CVPR_ , 2019. 2, 8 

- [77] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, 

Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _ICML_ , 2021. 4, 17 

- [78] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _JMLR_ , 2020. 1, 5 

- [79] Tanzila Rahman, Bicheng Xu, and Leonid Sigal. Watch, listen and tell: Multi-modal weakly supervised dense event captioning. In _ICCV_ , 2019. 2 

- [80] Paul Hongsuck Seo, Arsha Nagrani, Anurag Arnab, and Cordelia Schmid. End-to-end generative pretraining for multimodal video captioning. In _CVPR_ , 2022. 1, 2, 6, 8, 17 

- [81] Paul Hongsuck Seo, Arsha Nagrani, and Cordelia Schmid. Look before you speak: Visually contextualized utterances. In _CVPR_ , 2021. 2 

- [82] Zhiqiang Shen, Jianguo Li, Zhou Su, Minjun Li, Yurong Chen, Yu-Gang Jiang, and Xiangyang Xue. Weakly supervised dense video captioning. In _CVPR_ , 2017. 2 

- [83] Botian Shi, Lei Ji, Yaobo Liang, Nan Duan, Peng Chen, Zhendong Niu, and Ming Zhou. Dense procedure captioning in narrated instructional videos. In _ACL_ , 2019. 2 

- [84] Zheng Shou, Dongang Wang, and Shih-Fu Chang. Temporal action localization in untrimmed videos via multi-stage cnns. In _CVPR_ , 2016. 2 

- [85] Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela. FLAVA: A foundational language and vision alignment model. In _CVPR_ , 2022. 2 

- [86] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. _JMLR_ , 2014. 15 

- [87] Andreas Steiner, Alexander Kolesnikov, Xiaohua Zhai, Ross Wightman, Jakob Uszkoreit, and Lucas Beyer. How to train your vit? data, augmentation, and regularization in vision transformers. In _TMLR_ , 2022. 17 

- [88] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. VL-BERT: Pre-training of generic visual-linguistic representations. In _ICLR_ , 2019. 2 

- [89] Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid. VideoBERT: A joint model for video and language representation learning. In _ICCV_ , 2019. 2 

- [90] Yuchong Sun, Hongwei Xue, Ruihua Song, Bei Liu, Huan Yang, and Jianlong Fu. Long-form video-language pretraining with multimodal temporal contrastive learning. In _NeurIPS_ , 2022. 2 

- [91] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In _CVPR_ , 2016. 15 

- [92] Hao Tan and Mohit Bansal. LXMERT: Learning crossmodality encoder representations from transformers. In _EMNLP_ , 2019. 2 

- [93] Maria Tsimpoukelli, Jacob Menick, Serkan Cabi, SM Eslami, Oriol Vinyals, and Felix Hill. Multimodal few-shot learning with frozen language models. In _NeurIPS_ , 2021. 2 

11 

- [94] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _NeurIPS_ , 2017. 1, 4 

- [95] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. CIDEr: Consensus-based image description evaluation. In _CVPR_ , 2015. 6 

- [96] Alex Jinpeng Wang, Yixiao Ge, Rui Yan, Yuying Ge, Xudong Lin, Guanyu Cai, Jianping Wu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. All in one: Exploring unified video-language pre-training. _arXiv preprint arXiv:2203.07303_ , 2022. 2 

- [97] Bairui Wang, Lin Ma, Wei Zhang, and Wei Liu. Reconstruction network for video captioning. In _CVPR_ , 2018. 2 

- [98] Jinpeng Wang, Yixiao Ge, Guanyu Cai, Rui Yan, Xudong Lin, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Object-aware video-language pre-training for retrieval. In _CVPR_ , 2022. 2 

- [99] Jianfeng Wang, Xiaowei Hu, Zhe Gan, Zhengyuan Yang, Xiyang Dai, Zicheng Liu, Yumao Lu, and Lijuan Wang. UFO: A unified transformer for vision-language representation learning. _arXiv preprint arXiv:2111.10023_ , 2021. 2 

- [100] Jingwen Wang, Wenhao Jiang, Lin Ma, Wei Liu, and Yong Xu. Bidirectional attentive fusion with context gating for dense video captioning. In _CVPR_ , 2018. 1, 2 

- [101] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In _ICML_ , 2022. 3 

- [102] Teng Wang, Ruimao Zhang, Zhichao Lu, Feng Zheng, Ran Cheng, and Ping Luo. End-to-end dense video captioning with parallel decoding. In _ICCV_ , 2021. 1, 2, 7, 8, 14 

- [103] Teng Wang, Huicheng Zheng, Mingjing Yu, Qian Tian, and Haifeng Hu. Event-centric hierarchical representation for dense video captioning. _IEEE Transactions on Circuits and Systems for Video Technology_ , 2020. 2, 7 

- [104] Xin Wang, Wenhu Chen, Jiawei Wu, Yuan-Fang Wang, and William Yang Wang. Video captioning via hierarchical reinforcement learning. In _CVPR_ , 2018. 2 

- [105] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. SimVLM: Simple visual language model pretraining with weak supervision. In _ICLR_ , 2022. 1 

- [106] Zixu Wang, Yujie Zhong, Yishu Miao, Lin Ma, and Lucia Specia. Contrastive video-language learning with finegrained frame sampling. In _AACL-IJCNLP_ , 2022. 2 

- [107] Yilei Xiong, Bo Dai, and Dahua Lin. Move forward and tell: A progressive generator of video descriptions. In _ECCV_ , 2018. 8 

- [108] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. VideoCLIP: Contrastive pretraining for zero-shot video-text understanding. In _EMNLP_ , 2021. 2 

- [109] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. MSR-VTT: A large video description dataset for bridging video and language. In _CVPR_ , 2016. 2, 6, 14 

- [110] Mengmeng Xu, Erhan Gundogdu, Maksim Lapin, Bernard Ghanem, Michael Donoser, and Loris Bazzani. Contrastive language-action pre-training for temporal localization. _arXiv preprint arXiv:2204.12293_ , 2022. 2 

- [111] Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Advancing high-resolution video-language representation with large-scale video transcriptions. In _CVPR_ , 2022. 2 

- [112] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Just ask: Learning to answer questions from millions of narrated videos. In _ICCV_ , 2021. 2 

- [113] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Learning to answer visual questions from web videos. _IEEE TPAMI_ , 2022. 2 

- [114] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. TubeDETR: Spatio-temporal video grounding with transformers. In _CVPR_ , 2022. 2 

- [115] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Zero-shot video question answering via frozen bidirectional language models. In _NeurIPS_ , 2022. 2 

- [116] Jianwei Yang, Yonatan Bisk, and Jianfeng Gao. Taco: Token-aware cascade contrastive learning for video-text alignment. In _ICCV_ , 2021. 2 

- [117] Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Faisal Ahmed, Zicheng Liu, Yumao Lu, and Lijuan Wang. Crossing the format boundary of text and boxes: Towards unified vision-language modeling. In _ECCV_ , 2021. 1, 3 

- [118] Fei Yu, Jiji Tang, Weichong Yin, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. ERNIE-ViL: Knowledge enhanced vision-language representations through scene graph. In _AAAI_ , 2020. 2 

- [119] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. _Transactions on Machine Learning Research_ , 2022. 2 

- [120] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A new foundation model for computer vision. _arXiv preprint arXiv:2111.11432_ , 2021. 2 

- [121] Rowan Zellers, Jiasen Lu, Ximing Lu, Youngjae Yu, Yanpeng Zhao, Mohammadreza Salehi, Aditya Kusupati, Jack Hessel, Ali Farhadi, and Yejin Choi. MERLOT Reserve: Neural script knowledge through vision and language and sound. In _CVPR_ , 2022. 2, 3, 5, 6, 7, 14 

- [122] Rowan Zellers, Ximing Lu, Jack Hessel, Youngjae Yu, Jae Sung Park, Jize Cao, Ali Farhadi, and Yejin Choi. MERLOT: Multimodal neural script knowledge models. In _NeurIPS_ , 2021. 2, 6 

- [123] Chenlin Zhang, Jianxin Wu, and Yin Li. Actionformer: Localizing moments of actions with transformers. In _ECCV_ , 2022. 8 

- [124] Haotian Zhang, Pengchuan Zhang, Xiaowei Hu, Yen-Chun Chen, Liunian Harold Li, Xiyang Dai, Lijuan Wang, Lu Yuan, Jenq-Neng Hwang, and Jianfeng Gao. GLIPv2: Unifying localization and vision-language understanding. In _NeurIPS_ , 2022. 2 

12 

- [125] Qi Zhang, Yuqing Song, and Qin Jin. Unifying event detection and captioning as sequence generation via pre-training. In _ECCV_ , 2022. 2, 7 

- [126] Ziqi Zhang, Yaya Shi, Chunfeng Yuan, Bing Li, Peijin Wang, Weiming Hu, and Zheng-Jun Zha. Object relational graph with teacher-recommended learning for video captioning. In _CVPR_ , 2020. 8 

- [127] Yue Zhao, Yuanjun Xiong, Limin Wang, Zhirong Wu, Xiaoou Tang, and Dahua Lin. Temporal action detection with structured segment networks. In _ICCV_ , 2017. 2 

- [128] Luowei Zhou, Yannis Kalantidis, Xinlei Chen, Jason J Corso, and Marcus Rohrbach. Grounded video description. In _CVPR_ , 2019. 2, 8, 14 

- [129] Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason J Corso, and Jianfeng Gao. Unified vision-language pre-training for image captioning and VQA. In _AAAI_ , 2020. 2 

- [130] Luowei Zhou, Chenliang Xu, and Jason J Corso. Towards automatic learning of procedures from web instructional videos. In _AAAI_ , 2018. 1, 2, 4, 5, 6, 14 

- [131] Luowei Zhou, Yingbo Zhou, Jason J Corso, Richard Socher, and Caiming Xiong. End-to-end dense video captioning with masked transformer. In _CVPR_ , 2018. 1, 2, 7, 8 

- [132] Chaoyang Zhu, Yiyi Zhou, Yunhang Shen, Gen Luo, Xingjia Pan, Mingbao Lin, Chao Chen, Liujuan Cao, Xiaoshuai Sun, and Rongrong Ji. SeqTR: A simple yet universal network for visual grounding. In _ECCV_ , 2022. 3 

- [133] Wanrong Zhu, Bo Pang, Ashish Thapliyal, William Yang Wang, and Radu Soricut. End-to-end dense video captioning as sequence generation. In _COLING_ , 2022. 2, 3, 7 

13 

## **Appendix** 

In this Appendix, we present the following additional material: 

- _(i)_ Additional qualitative examples of dense video captioning predictions (Section A). 

- _(ii)_ Additional information about our experimental setup (Section B); 

- _(iii)_ Additional experimental results (Section C), including an ablation on the importance of pretraining for fewshot dense video captioning (Section C.1) and additional ablation studies in the standard fully-supervised dense video captioning setting (Section C.2). 

## **A. Qualitative examples of dense video captioning predictions** 

In Figure 4, we show qualitative results of dense event captioning by our Vid2Seq model. Here in Figures 5 and 6 , we show additional results on examples from the YouCook2 and ActivityNet Captions datasets. These results show that Vid2Seq can predict meaningful dense captions and event boundaries in diverse scenarios, with or without transcribed speech input, _e.g_ . series of instructions in cooking recipes (Figure 5) or actions in human sports or leisure activities (first three examples in Figure 6). The last example in Figure 6 illustrates a failure case where the model hallucinates events that are not visually grounded such as ‘one man hats off to the camera‘. 

## **B. Experimental setup** 

In this section, we complement the information provided in Section 4.1 about the datasets we use (Section B.1). We also give additional implementation details (Section B.2). 

## **B.1. Datasets** 

**YT-Temporal-1B** [121] consists of 18.821M unlabeled narrated videos covering about 150 years of video content for pretraining. Compared with HowTo100M [72], this dataset was created to cover a wider range of domains and not only instructional videos. 

**HowTo100M** [72] consists of 1.221M unlabeled narrated instructional videos covering about 15 years of video content for pretraining. 

**YouCook2** [130] has 1,790 untrimmed videos of cooking procedures. On average, each video lasts 320s and is annotated with 7.7 temporally-localized imperative sentences. The dataset is split into 1,333 videos for training and 457 videos for validation. 

**ViTT** [35] consists of 7,672 untrimmed instructional videos from the YouTube-8M dataset [2]. Compared to YouCook2, 

ViTT was created to better reflect the distribution of instructional videos in the wild. On average, each video lasts 250s and is annotated with 7.1 temporally-localized short tags. The dataset is split into 5,476, 1,102 and 1,094 videos for training, validation and testing, respectively. Videos in the validation and test sets are provided with multiple sets of dense event captioning annotations. Following [35], we treat each set of annotations as a single example during evaluation and discard videos with more than 3 sets of annotations. 

**ActivityNet-Captions** [46] contains 14,934 untrimmed videos of various human activities. Different from YouCook2 and ViTT where most videos contain transcribed speech content, we find that 68% of videos in ActivityNet Captions do not have transcribed narration. On average, each video lasts 120s and is annotated with 3.7 temporally-localized sentences. The dataset is split into 10,009 and 4,925 videos for training and validation, respectively. Videos in the validation set are provided with two sets of dense video captioning annotations. Following prior work [102], we use both sets of annotations for evaluation, by computing the average of the scores over each set for SODA ~~c~~ and by using the standard evaluation tool [46] for all other dense event captioning metrics. For video paragraph captioning, we follow [102] and report results on the ’val-ae’ split that includes 2,460 videos [50, 128]. 

**MSR-VTT** [109] consists of 10,000 open domain video clips. The duration of each video clip is between 10 and 30 seconds. 20 natural language descriptions are manually annotated for each clip. The dataset is split into 6,513, 497 and 2,990 videos for training, validation and testing, respectively. 

**MSVD** [9] consists of 1,970 open domain video clips. The duration of each video clip is between 10 and 30 seconds. Each video clip has roughly 40 manually annotated captions. The dataset is split into 1,200, 100 and 670 videos for training, validation and testing, respectively. 

## **B.2. Implementation details** 

**Architecture.** The visual temporal transformer encoder _f[t]_ , the text encoder _g[t]_ and the text decoder _h[t]_ all have 12 layers, 12 heads, embedding dimension 768, and MLP hidden dimension of 2048. The text encoder and decoder sequences are truncated or padded to _L_ = _S_ = 1000 tokens during pretraining, and _S_ = 1000 and _L_ = 256 tokens during finetuning. At inference, we use beam search decoding where we track the top 4 sequences and apply a length normalization of 0.6. 

**Training.** We use the Adam optimizer [42] with _β_ = (0 _._ 9 _,_ 0 _._ 999) and no weight decay. During pretraining, we use a learning rate of 1 _e[−]_[4] , warming it up linearly (from 

14 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Vid2Seq_Large-Scale_Pretraining_of_a_Visual_Language_Model_for_Dense_Video_Captioning_images/cvpr_2023_Vid2Seq_Large-Scale_Pretraining_of_a_Visual_Language_Model_for_Dense_Video_Captioning.pdf-0015-00.png)


**----- Start of picture text -----**<br>
I'm going I'm just I've got a piece The  So I'm Now, I'm I'm just Now,  I've got my small We're And if you'd<br>to start  going to  of wax paper first  going to  using my going to  the  cast-iron skillet on  going to  like to<br>off with trim off  here and I put  thing take two homema mix this breading medium-high heat be baking follow me<br>SpeechInput  twobonelessskinless the grislyparts and the  … that onto mycutting board […]  and I'm … I'mgoingto  large eggsand crack those into … de Italianbreadcrumbs … togetherand nowwe can  process is reallysimple  here and I'm going to put in about a quarter of an inch or so of  these and that willfinish  … on Google Plus Facebook<br>chicken excess going to pound  need is a bowl  here. start  on this extra virgin olive oil cooking  and/or<br>breasts fat  out my breast an egg and if you breading you just into the bottom of  them. Pinterest all<br>here. maybe halves until wash. get any our want to  that and I'm going to  my links<br>some of  they are about  shells in  chicken. take one  let that come up to  will be in<br>the skin  1/2 an inch there, be of your temperature and then the<br>that's left thicker. sure to get […]  I'm going to start  description<br>over on  those  […] frying up my chicken box.<br>there. pieces.<br>Input<br>Frames<br>GT Cut the chicken. Pound the  Whisk the eggs. Mix bread Mix  Coat the  Add Fry the  Place  Add marinara  Bake the<br>chicken. crumbs flour chicken in the  oil to  chicken the  sauce and  chicken in<br>and  salt and  flour mixture  a pan. in the  chicken cheese on top  an oven.<br>parmesan  pepper the egg pan. in a  of the chicken.<br>cheese together mixture and  baking<br>together. . then the bread dish.<br>crumbs.<br>Vis2Seq Trim off the  Cover the chicken in  Crack two large  Add bread crumbs Coat the chicken in  Fry the  Pour tomato sauce  Bake the chicken in an<br>excess fat of  plastic wrap and  eggs into a bowl  grated parmesan  the flour mixture  chicken in a  and mozzarella  oven.<br>chicken breast and  pound it out. and whisk them cheese and italian and then the bread pan with cheese on top of the<br>cut it into halves. together. bread crumbs to a  crumbs. oil. chicken.<br>bowl.<br>And  There  So then So the  So in  If you do have  So now So it's gonna So no  And after<br>today we go. we're rule of  this fresh dill, I  we're going be really well oxygen that it can be<br>what I  going thumb case,  would to get in  covered and  that's popped into<br>want to  to chop  with salt we've recommend really mix  obviously it's that's the fridge or<br>SpeechInput  do isshare one of my … the cabbage you … is that youwantabout 2.5  gotabout 20  … using that nowfor the fun part, what you … that salt intothe cabbagewhich is … important to wash yourhands before … the idea. … popped intoanother jar and enjoyed,<br>favorite  can use  to 3  grams. want to do is going to take you do this you know, as<br>recipes a  percent of  we really need about ten that seems a condiment<br>and in  mandol the weight to release all  minutes so like a pretty once a day<br>my in to do  of the  of the juices really work obvious thing […]<br>opinion this. cabbage. from this it it's a good  to do but  […]<br>[…] cabbage workout.<br>Input<br>Frames<br>GT Finely chop a  Add 20g salt Massage the cabbage Put the mixture in a  Seal the jar<br>cabbage to small caraway seeds with the seasoning. jar and press firm to  and put in<br>pieces. juniper berries the bottom. dark place<br>and dill. for 4 weeks.<br>Vis2Seq Take a large cabbage Add some salt to  Add caraway seeds juniper Place the  Seal the<br>into a bowl and chop  the cabbage and  berries and dill tips to the  mixture  jar.<br>it into small pieces. mix. cabbage and mix. into a<br>glass jar.<br>… …<br>**----- End of picture text -----**<br>


Figure 5. Examples of dense event captioning predictions of Vid2Seq on the validation set of YouCook2, compared with ground-truth. 

0) for the first 1000 iterations, and keeping it constant for the remaining iterations. During finetuning, we use a learning rate of 3 _e−_ 4, warming it up linearly (from 0) for the first 10% of iterations, followed by a cosine decay (down to 0) for the remaining 90%. During finetuning, we use a batch size of 32 videos split on 16 TPU v4 chips. We finetune for 40 epochs on YouCook2, 20 epochs on ActivityNet Captions and ViTT, 5 epochs on MSR-VTT and 10 epochs on MSVD. We clip the maximum norm of the gradient to 0.1 during pretraining, and 1 during finetuning. For data 

augmentation, we use random temporal cropping. For regularization, we use label smoothing [91] with value 0.1 and dropout [86] with probability 0.1. 

## **C. Experiments** 

In this section, we provide additional experiments that complement the results presented in Section 4. We first show the importance of pretraining in our proposed fewshot setting in Section C.1. Then we provide additional 

15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Vid2Seq_Large-Scale_Pretraining_of_a_Visual_Language_Model_for_Dense_Video_Captioning_images/cvpr_2023_Vid2Seq_Large-Scale_Pretraining_of_a_Visual_Language_Model_for_Dense_Video_Captioning.pdf-0016-00.png)


**----- Start of picture text -----**<br>
Input  Ø Ø Ø Ø Ø Ø Ø Ø<br>Speech<br>Input<br>Frames<br>GT A weightlifter isstanding on a  He lifts the barbell before dropping it. He jumps up and down in excitement.<br>stage.<br>Vis2Seq A very strong He lifts a very heavy weight over his head. He then drops the weight to the ground before shaking<br>man is shown in a  his hands.<br>competition<br>Input  Ø Ø Ø Ø Ø Ø Ø Ø<br>Speech<br>Input<br>Frames<br>GT A man walks up to parallel bars  The man performs a routine on the parallel bars. The man finishes his routine and dismounts.<br>while spectators,<br>competitors, and<br>officials are in the<br>background.<br>Vis2Seq A man walks up to a set of uneven bars. He mounts the bars, then spins himself around. He does a handstand, then dismounts.<br>Input  Ø Ø Ø Ø Ø Ø Ø Ø<br>Speech<br>Input<br>Frames<br>GT A man is seen looking at the camera and leads intohim playing a poker game with others. One man deals cards and chips whilespeaking to one another. They continue playing and speaking to one another.<br>Vis2Seq A man is sitting behind a table playing He deals cards to the people, then he puts them on the table. The man puts the<br>poker.<br>cards on the table,<br>and puts the chips<br>in the middle.<br>Input  Ø Ø Ø Ø Ø Ø Ø Ø<br>Speech<br>Input<br>Frames<br>GT A group of children are seenswimming in a pool. The kids hit a ball back and forth in the water. They fight over the ball, trying to get it into the goal.<br>A picture of a sky is shown and leads into a  The camera pans around a small group of kids playing and then a man  The boys continue playing and one man hats off to the<br>Vis2Seq group of boys playing a game of water polo. chases a ball around. camera.<br>**----- End of picture text -----**<br>


Figure 6. Examples of dense event captioning predictions by Vid2Seq on the validation set of ActivityNet Captions, compared with groundtruth. The first three examples show successful predictions, while the last example illustrates a failure case where the model hallucinates events that are not visually grounded (‘one man hats off to the camera‘). Note that in all of these videos, there is no transcribed speech. 

ablation studies in the standard fully-supervised setting in Section C.2, where we ablate various factors including pretraining on long narrated videos, the pretraining dataset and the size of the visual backbone, the time tokenization process and the number of time tokens, the sequence construction process, the temporal positional embeddings and the 

initialization of the language model. 

## **C.1. Importance of pretraining in few-shot settings** 

In Section 4.2, we show the benefits of our pretraining method in the fully-supervised setting, _i.e_ . when using 100% of the downstream training dataset. In Table 10, we 

16 

|Data<br>Pretrain|YouCook2<br>S<br>C<br>M|ViTT<br>S<br>C<br>M|ActivityNet<br>S<br>C<br>M|
|---|---|---|---|
|1.<br>1%<br>✗<br>2.<br>1%<br>✓<br>3.<br>10%<br>✗<br>4.<br>10%<br>✓<br>5.<br>50%<br>✗<br>6.<br>50%<br>✓<br>7. 100%<br>✗<br>8. 100%<br>✓|0.0<br>0.0<br>0.0<br>2.4 10.1 3.3<br>0.1<br>0.0<br>0.2<br>3.8 18.4 5.2<br>1.8<br>8.5<br>2.4<br>6.2 32.1 7.6<br>4.0 18.0 4.6<br>**7.9 47.1 9.3**|0.0<br>0.0<br>0.0<br>2.0<br>7.4<br>1.9<br>3.3<br>0.4<br>3.3<br>10.7 28.6<br>6.0<br>6.5<br>18.7<br>3.9<br>12.5 38.8<br>7.8<br>7.9<br>21.2<br>6.2<br>**13.5 43.5**<br>**8.5**|0.0<br>0.0<br>0.1<br>2.2<br>6.2<br>3.2<br>3.4<br>11.9 4.6<br>4.3<br>20.0 6.1<br>4.6<br>13.1 6.3<br>5.4<br>27.5 7.8<br>5.4<br>18.8 7.1<br>**5.8**<br>**30.1 8.5**|



Table 10. **Impact of our pretraining on few-shot dense event captioning** , by finetuning Vid2Seq using a small fraction of the downstream training dataset. 

|Max number<br>of narrations|YouCook2<br>S<br>C<br>F1|ActivityNet<br>S<br>C<br>F1|
|---|---|---|
|1.<br>_No pretraining_<br>2.<br>1<br>3.<br>10<br>4.<br>_∞_|4.0<br>18.0<br>18.1<br>6.0<br>32.1<br>22.1<br>6.5<br>34.6<br>23.6<br>**7.9**<br>**47.1**<br>**27.3**|5.4<br>18.8<br>49.2<br>5.1<br>22.9<br>48.1<br>5.4<br>27.1<br>50.3<br>**5.8**<br>**30.1**<br>**52.4**|



Table 11. **Ablation showing the importance of pretraining on long narrated videos** , by varying the maximum number of narration sentences that a randomly cropped video can cover. _∞_ means the cropping is unrestricted and can sample arbitrarily long videos. 

|Pretraining Data<br>Model|YouCook2<br>S<br>C<br>F1|ActivityNet<br>S<br>C<br>F1|
|---|---|---|
|1.<br>ImageNet<br>ViT-B/16<br>2.<br>CLIP<br>ViT-B/16<br>3.<br>CLIP<br>ViT-L/14|6.6<br>40.2<br>24.3<br>7.7<br>46.3<br>26.5<br>**7.9**<br>**47.1**<br>**27.3**|4.5<br>17.2<br>49.3<br>5.6<br>28.4<br>51.7<br>**5.8**<br>**30.1**<br>**52.4**|



Table 12. **Ablation on the pretraining data and model size of the visual backbone** _f[s]_ **.** 

further show that our pretraining method has a considerable importance in the few-shot setting defined in Section 4.4, _i.e_ . when using a smaller fraction of the downstream training dataset. In particular, our pretraining method enables our Vid2Seq model to have a non zero performance when using only 1% of the downstream training dataset (rows 1 and 2). 

## **C.2. Additional ablation studies** 

We here complement ablation studies reported in Section 4.2, using the same default settings, evaluation metrics and downstream datasets. 

**Pretraining on long narrated videos.** In Table 1, we show the benefits of pretraining on untrimmed videos in comparison with the standard practice of pretraining on short, trimmed, video-speech segments [35, 70, 80]. In Table 11, we further evaluate the importance of sampling long narrated videos during pretraining. By default, at each training iteration, we randomly temporally crop each narrated video without constraints, resulting in a video that can span 

|Tokenization<br>_N_|YouCook2<br>S<br>C<br>F1|ActivityNet<br>S<br>C<br>F1|
|---|---|---|
|1.<br>Absolute<br>20<br>0.3<br>0.2<br>0.9<br>3.2<br>23.0<br>23.1<br>2.<br>Absolute<br>100<br>3.5<br>25.7<br>12.0<br>4.8<br>25.5<br>41.5<br>3.<br>Absolute<br>500<br>**7.9**<br>39.8<br>24.3<br>5.4<br>28.1<br>48.6<br>4.<br>Relative<br>20<br>7.2<br>39.6<br>23.7<br>5.6<br>29.0<br>49.4<br>5.<br>Relative<br>100<br>**7.9**<br>**47.1**<br>**27.3**<br>**5.8**<br>**30.1**<br>52.4<br>6.<br>Relative<br>500<br>7.2<br>40.0<br>25.0<br>5.7<br>28.6<br>**52.5**|||



Table 13. **Ablation on time tokenization (relative or absolute) and the number of time tokens** _N_ **.** 

|Dot symbol<br>between segments<br>Time tokens<br>Position|YouCook2<br>S<br>C<br>F1|ActivityNet<br>S<br>C<br>F1|
|---|---|---|
|1.<br>✗<br>_After text_<br>2.<br>✓<br>_After text_<br>3.<br>✗<br>_Before text_<br>4.<br>✓<br>_Before text_|7.9<br>48.3<br>26.7<br>**8.3**<br>**50.9**<br>26.2<br>8.0<br>50.0<br>**27.3**<br>7.9<br>47.1<br>**27.3**|5.6<br>29.8<br>51.1<br>5.7 **30.4**<br>51.8<br>5.6<br>28.2<br>50.7<br>**5.8** 30.1<br>**52.4**|



Table 14. **Ablation on the sequence construction process.** 

over hundreds of transcribed speech sentences. We here evaluate a baseline that constrains this cropping process such that the cropped video only spans over a given maximum number of narration sentences. Even with a maximum of 10 narration sentences, this baseline significantly underperforms our model trained in default settings where we sample longer untrimmed narrated videos (rows 1, 2 and 3). This demonstrates that our model benefits from pretraining on long narrated videos. 

**Visual features.** In Table 4, we show the benefits of scaling up the size of the pretraining dataset of narrated videos and the size of the language model. In Table 12, we further analyze the importance of the pretraining dataset and size of the visual backbone _f[s]_ . We find that CLIP pretraining [77] considerably improves over ImageNet pretraining [87] with the same ViT-B/16 visual backbone model (row 2 vs 1). Furthermore, scaling up the visual backbone size from ViTB/16 to ViT-L/14 brings additional improvements (row 3 vs 2). 

**Time tokenization and number of time tokens.** In Table 13, we further ablate the time tokenization process presented in Section 3.1. Our default time tokens represent relative timestamps in a video, as we quantize a video of duration _T_ into _N_ equally-spaced timestamps. Another possibility is to use time tokens that represent absolute timestamps in the video, _i.e_ . the k-th token represents the k-th second in the video. For both these variants, we vary the number of time tokens _N_ . For the relative time tokens, increasing _N_ makes the quantization more fine-grained but also spreads the data into more time tokens. On the other hand, for the absolute time tokens, increasing _N_ increases the video du- 

17 

|Temporal<br>embeddings|YouCook2<br>S<br>C<br>F1|ActivityNet<br>S<br>C<br>F1|
|---|---|---|
|1.<br>✗<br>2.<br>✓|6.8<br>42.0<br>24.9<br>**7.9**<br>**47.1**<br>**27.3**|5.3<br>27.0<br>50.6<br>**5.8**<br>**30.1**<br>**52.4**|



of the domain gap between Web text and the imperativestyle dense captions in YouCook2, which are more similar to transcribed speech in YT-Temporal-1B. 

Table 15. **Ablation on the temporal positional embeddings.** 

|Language Model<br>Initialization<br>Video-text<br>Pretraining|YouCook2<br>S<br>C<br>F1|ActivityNet<br>S<br>C<br>F1|
|---|---|---|
|1.<br>✗<br>✗<br>2.<br>✓<br>✗<br>3.<br>✗<br>✓<br>4.<br>✓<br>✓|0.9<br>4.2<br>7.6<br>4.0<br>18.0<br>18.1<br>**8.8**<br>**51.3**<br>**28.4**<br>7.9<br>47.1<br>27.3|4.3<br>23.7<br>41.2<br>5.4<br>18.8<br>49.2<br>5.7<br>28.7<br>51.2<br>**5.8**<br>**30.1**<br>**52.4**|



Table 16. **Ablation on language model initialization and pretraining.** 

ration that the time tokens can cover. We find that the best dense video captioning results are obtained with the relative time tokens and _N_ = 100 time tokens (row 5). 

**Sequence construction.** In Table 14, we further ablate the sequence construction process presented in Section 3.1. Our default sequence inserts the start and end time tokens of each segment before its corresponding text sentence. Another possibility is to insert time tokens after each corresponding text sentence. We find that both variants achieve similar results (rows 2 and 4), with the default sequence (row 4) resulting in slightly higher event localization performance (F1 Score) but slightly lower dense captioning results overall. Furthermore, we observe that the dot symbols indicating the separation between different events have low importance (rows 1 and 2, rows 3 and 4). 

**Temporal positional embeddings.** In Table 1, we show that time tokens in the speech sequence provide temporal information about the speech transcript to our model. In Table 15, we also evaluate the importance of the temporal positional embeddings which communicate temporal information from the visual stream to our model. We find that these temporal embeddings are beneficial (row 2 vs 1). 

**Language model initialization and pretraining.** In Table 4, we show the benefits of using T5-Base instead of T5-Small. In Table 16, we further investigate the importance of initializing the language model from weights pretrained on Web text. Without pretraining on narrated videos, we find that text-only initialization is helpful (rows 1 and 2). Interestingly, after pretraining on narrated videos, we find that text-only initialization has little importance (rows 3 and 4), as it slightly improves the performance on ActivityNet Captions while resulting in a slight drop of performance on YouCook2. We believe that this may be because 

18 

