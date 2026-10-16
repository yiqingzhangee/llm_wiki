```
SpeechandLanguageProcessing.DanielJurafsky&JamesH.Martin.Copyright©2021.All
rightsreserved.DraftofDecember29,2021.
```

CHAPTER 

## 25 **Phonetics** 

The characters that make up the texts we’ve been discussing in this book aren’t just random symbols. They are also an amazing scientific invention: a theoretical model of the elements that make up human speech. 

The earliest writing systems we know of (Sumerian, Chinese, Mayan) were mainly _logographic_ : one symbol representing a whole word. But from the earliest stages we can find, some symbols were also used to represent the sounds that made up words. The cuneiform sign to the right pronounced _ba_ and meaning “ration” in Sumerian could also function purely as the sound /ba/. The earliest Chinese characters we have, carved into bones for divination, similarly contain phonetic elements. Purely sound-based writing systems, whether syllabic (like Japanese _hiragana_ ), alphabetic (like the Roman alphabet), or consonantal (like Semitic writing systems), trace back to these early logo-syllabic systems, often as two cultures came together. Thus, the Arabic, Aramaic, Hebrew, Greek, and Roman systems all derive from a West Semitic script that is presumed to have been modified by Western Semitic mercenaries from a cursive form of Egyptian hieroglyphs. The Japanese syllabaries were modified from a cursive form of Chinese phonetic characters, which themselves were used in Chinese to phonetically represent the Sanskrit in the Buddhist scriptures that came to China in the Tang dynasty. 

This implicit idea that the spoken word is composed of smaller units of speech underlies algorithms for both **speech recognition** (transcribing waveforms into text) and **text-to-speech** (converting text into waveforms). In this chapter we give a com- **phonetics** putational perspective on **phonetics** , the study of the speech sounds used in the languages of the world, how they are produced in the human vocal tract, how they are realized acoustically, and how they can be digitized and processed. 

## 25.1 Speech Sounds and Phonetic Transcription 

A letter like ‘p’ or ‘a’ is already a useful model of the sounds of human speech, and indeed we’ll see in Chapter 26 how to map between letters and waveforms. Nonetheless, it is helpful to represent sounds slightly more abstractly. We’ll repre- **phone** sent the pronunciation of a word as a string of **phones** , which are speech sounds, each represented with symbols adapated from the Roman alphabet. 

The standard phonetic representation for transcribing the world’s languages is **IPA** the **International Phonetic Alphabet** ( **IPA** ), an evolving standard first developed in 1888, But in this chapter we’ll instead represent phones with the ARPAbet (Shoup, 1980), a simple phonetic alphabet (Fig. 25.1) that conveniently uses ASCII symbols to represent an American-English subset of the IPA. 

Many of the IPA and ARPAbet symbols are equivalent to familiar Roman letters. So, for example, the ARPAbet phone [p] represents the consonant sound at the 

**2** CHAPTER 25 • PHONETICS 

|**ARPAbet IPA**<br>**Symbol**<br>**Symbol **|**ARPAbet**<br>**ARPAbet IPA**<br>**ARPAbet**<br> **Word**<br>**Transcription**<br>**Symbol**<br>**Symbol Word Transcription**<br>p<br>arsley<br>[p aa r s l iy]<br>[iy]<br>[i]<br>lily<br>[l ih l iy]<br>t<br>ea<br>[t iy]<br>[ih]<br>[I]<br>li<br>ly<br>[l ih l iy]<br>c<br>ook<br>[k uh k]<br>[ey]<br>[eI]<br>dai<br>sy<br>[d ey z iy]<br>b<br>ay<br>[b ey]<br>[eh]<br>[E]<br>pe<br>n<br>[p eh n]<br>d<br>ill<br>[d ih l]<br>[ae]<br>[æ]<br>a<br>ster<br>[ae s t axr]<br>g<br>arlic<br>[g aa r l ix k]<br>[aa]<br>[A]<br>po<br>ppy [p aa p iy]<br>m<br>int<br>[m ih n t]<br>[ao]<br>[O]<br>o<br>rchid [ao r k ix d]<br>n<br>utmeg<br>[n ah t m eh g]<br>[uh]<br>[U]<br>woo<br>d<br>[w uh d]<br>baking<br>[b ey k ix ng]<br>[ow]<br>[oU]<br>lo<br>tus<br>[l ow dx ax s]<br>f<br>lour<br>[f l aw axr]<br>[uw]<br>[u]<br>tu<br>lip<br>[t uw l ix p]<br>clov<br>e<br>[k l ow v]<br>[ah]<br>[2]<br>bu<br>tter<br>[b ah dx axr]<br>th<br>ick<br>[th ih k]<br>[er]<br>[Ç]<br>bir<br>d<br>[b er d]<br>th<br>ose<br>[dh ow z]<br>[ay]<br>[aI]<br>i<br>ris<br>[ay r ix s]<br>s<br>oup<br>[s uw p]<br>[aw]<br>[aU]<br>fow<br>er [f l aw axr]<br>eggs<br>[eh g z]<br>[oy]<br>[oI]<br>soi<br>l<br>[s oy l]<br>squash<br>[s k w aa sh]<br>ambros<br>ia [ae m b r ow zh ax]<br>ch<br>erry<br>[ch eh r iy]<br>j<br>ar<br>[jh aa r]<br>l<br>icorice<br>[l ih k axr ix sh]<br>kiw<br>i<br>[k iy w iy]<br>r<br>ice<br>[r ay s]<br>y<br>ellow<br>[y eh l ow]<br>h<br>oney<br>[h ah n iy]|
|---|---|
|[p]<br>[p]<br>[t]<br>[t]<br>[k]<br>[k]<br>[b]<br>[b]<br>[d]<br>[d]<br>[g]<br>[g]<br>[m]<br>[m]<br>[n]<br>[n]<br>[ng]<br>[N]<br>[f]<br>[f]<br>[v]<br>[v]<br>[th]<br>[T]<br>[dh]<br>[D]<br>[s]<br>[s]<br>[z]<br>[z]<br>[sh]<br>[S]<br>[zh]<br>[Z]<br>[ch]<br>[tS]<br>[jh]<br>[dZ]<br>[l]<br>[l]<br>[w]<br>[w]<br>[r]<br>[r]<br>[y]<br>[j]<br>[h]<br>[h]||



**Figure 25.1** ARPAbet and IPA symbols for English consonants (left) and vowels (right). 

beginning of _platypus_ , _puma_ , and _plantain_ , the middle of _leopard_ , or the end of _antelope_ . In general, however, the mapping between the letters of English orthography and phones is relatively **opaque** ; a single letter can represent very different sounds in different contexts. The English letter _c_ corresponds to phone [k] in _cougar_ [k uw g axr], but phone [s] in _cell_ [s eh l]. Besides appearing as _c_ and _k_ , the phone [k] can appear as part of _x_ ( _fox_ [f aa k s]), as _ck_ ( _jackal_ [jh ae k el]) and as _cc_ ( _raccoon_ [r ae k uw n]). Many other languages, for example, Spanish, are much more **transparent** in their sound-orthography mapping than English. 

## 25.2 Articulatory Phonetics 

> **articulatoryphonetics Articulatory phonetics** is the study of how these phones are produced as the various organs in the mouth, throat, and nose modify the airflow from the lungs. 

## **The Vocal Organs** 

Figure 25.2 shows the organs of speech. Sound is produced by the rapid movement of air. Humans produce most sounds in spoken languages by expelling air from the lungs through the windpipe (technically, the **trachea** ) and then out the mouth or nose. As it passes through the trachea, the air passes through the **larynx** , commonly known as the Adam’s apple or voice box. The larynx contains two small folds of 

25.2 • ARTICULATORY PHONETICS **3** 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0003-01.png)


**Figure 25.2** The vocal organs, shown in side view. (Figure from OpenStax University Physics, CC BY 4.0) 

**glottis** 

**voiced sound unvoiced sound** 

**nasal** 

**consonant vowel** 

muscle, the **vocal folds** (often referred to non-technically as the **vocal cords** ), which can be moved together or apart. The space between these two folds is called the **glottis** . If the folds are close together (but not tightly closed), they will vibrate as air passes through them; if they are far apart, they won’t vibrate. Sounds made with the vocal folds together and vibrating are called **voiced** ; sounds made without this vocal cord vibration are called **unvoiced** or **voiceless** . Voiced sounds include [b], [d], [g], [v], [z], and all the English vowels, among others. Unvoiced sounds include [p], [t], [k], [f], [s], and others. 

The area above the trachea is called the **vocal tract** ; it consists of the **oral tract** and the **nasal tract** . After the air leaves the trachea, it can exit the body through the mouth or the nose. Most sounds are made by air passing through the mouth. Sounds made by air passing through the nose are called **nasal sounds** ; nasal sounds (like English [m], [n], and [ng]) use both the oral and nasal tracts as resonating cavities. 

Phones are divided into two main classes: **consonants** and **vowels** . Both kinds of sounds are formed by the motion of air through the mouth, throat or nose. Consonants are made by restriction or blocking of the airflow in some way, and can be voiced or unvoiced. Vowels have less obstruction, are usually voiced, and are generally louder and longer-lasting than consonants. The technical use of these terms is much like the common usage; [p], [b], [t], [d], [k], [g], [f], [v], [s], [z], [r], [l], etc., are consonants; [aa], [ae], [ao], [ih], [aw], [ow], [uw], etc., are vowels. **Semivowels** (such as [y] and [w]) have some of the properties of both; they are voiced like vowels, but they are short and less syllabic like consonants. 

**4** CHAPTER 25 • PHONETICS 

## **Consonants: Place of Articulation** 

Because consonants are made by restricting airflow, we can group them into classes **articulationplace of** by their point of maximum restriction, their **place of articulation** (Fig. 25.3). 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0004-03.png)


**----- Start of picture text -----**<br>
(nasal tract)<br>alveolar<br>dental<br>palatal velar<br>bilabial<br>glottal<br>**----- End of picture text -----**<br>


**Figure 25.3** Major English places of articulation. 

> **labial Labial:** Consonants whose main restriction is formed by the two lips coming together have a **bilabial** place of articulation. In English these include [p] as in _possum_ , [b] as in _bear_ , and [m] as in _marmot_ . The English **labiodental** consonants [v] and [f] are made by pressing the bottom lip against the upper row of teeth and letting the air flow through the space in the upper teeth. 

> **dental Dental:** Sounds that are made by placing the tongue against the teeth are dentals. The main dentals in English are the [th] of _thing_ and the [dh] of _though_ , which are made by placing the tongue behind the teeth with the tip slightly between the teeth. 

> **alveolar Alveolar:** The alveolar ridge is the portion of the roof of the mouth just behind the upper teeth. Most speakers of American English make the phones [s], [z], [t], and [d] by placing the tip of the tongue against the alveolar ridge. The word **coronal** is often used to refer to both dental and alveolar. 

> **palatal Palatal:** The roof of the mouth (the **palate** ) rises sharply from the back of the 

> **palate** alveolar ridge. The **palato-alveolar** sounds [sh] ( _shrimp_ ), [ch] ( _china_ ), [zh] ( _Asian_ ), and [jh] ( _jar_ ) are made with the blade of the tongue against the rising back of the alveolar ridge. The palatal sound [y] of _yak_ is made by placing the front of the tongue up close to the palate. 

> **velar Velar:** The **velum** , or soft palate, is a movable muscular flap at the very back of the roof of the mouth. The sounds [k] ( _cuckoo_ ), [g] ( _goose_ ), and [N] ( _kingfisher_ ) are made by pressing the back of the tongue up against the velum. 

> **glottal Glottal:** The glottal stop [q] is made by closing the glottis (by bringing the vocal folds together). 

## **Consonants: Manner of Articulation** 

Consonants are also distinguished by _how_ the restriction in airflow is made, for example, by a complete stoppage of air or by a partial blockage. This feature is called **articulationmanner of** the **manner of articulation** of a consonant. The combination of place and manner of articulation is usually sufficient to uniquely identify a consonant. Following are the major manners of articulation for English consonants: **stop** A **stop** is a consonant in which airflow is completely blocked for a short time. This blockage is followed by an explosive sound as the air is released. The period of blockage is called the **closure** , and the explosion is called the **release** . English 

25.2 • ARTICULATORY PHONETICS **5** 

has voiced stops like [b], [d], and [g] as well as unvoiced stops like [p], [t], and [k]. Stops are also called **plosives** . 

**nasal** 

**fricatives** 

**sibilants** 

**approximant** 

**tap** 

The **nasal** sounds [n], [m], and [ng] are made by lowering the velum and allowing air to pass into the nasal cavity. 

In **fricatives** , airflow is constricted but not cut off completely. The turbulent airflow that results from the constriction produces a characteristic “hissing” sound. The English labiodental fricatives [f] and [v] are produced by pressing the lower lip against the upper teeth, allowing a restricted airflow between the upper teeth. The dental fricatives [th] and [dh] allow air to flow around the tongue between the teeth. The alveolar fricatives [s] and [z] are produced with the tongue against the alveolar ridge, forcing air over the edge of the teeth. In the palato-alveolar fricatives [sh] and [zh], the tongue is at the back of the alveolar ridge, forcing air through a groove formed in the tongue. The higher-pitched fricatives (in English [s], [z], [sh] and [zh]) are called **sibilants** . Stops that are followed immediately by fricatives are called **affricates** ; these include English [ch] ( _chicken_ ) and [jh] ( _giraffe_ ). 

In **approximants** , the two articulators are close together but not close enough to cause turbulent airflow. In English [y] ( _yellow_ ), the tongue moves close to the roof of the mouth but not close enough to cause the turbulence that would characterize a fricative. In English [w] ( _wood_ ), the back of the tongue comes close to the velum. American [r] can be formed in at least two ways; with just the tip of the tongue extended and close to the palate or with the whole tongue bunched up near the palate. [l] is formed with the tip of the tongue up against the alveolar ridge or the teeth, with one or both sides of the tongue lowered to allow air to flow over it. [l] is called a **lateral** sound because of the drop in the sides of the tongue. 

A **tap** or **flap** [dx] is a quick motion of the tongue against the alveolar ridge. The consonant in the middle of the word _lotus_ ([l ow dx ax s]) is a tap in most dialects of American English; speakers of many U.K. dialects would use a [t] instead. 

## **Vowels** 

Like consonants, vowels can be characterized by the position of the articulators as they are made. The three most relevant parameters for vowels are what is called vowel **height** , which correlates roughly with the height of the highest part of the tongue, vowel **frontness** or **backness** , indicating whether this high point is toward the front or back of the oral tract and whether the shape of the lips is **rounded** or not. Figure 25.4 shows the position of the tongue for different vowels. 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0005-13.png)


**----- Start of picture text -----**<br>
tongue palate<br>closed<br>velum<br>beet [iy] bat [ae] boot [uw]<br>**----- End of picture text -----**<br>


**Figure 25.4** Tongue positions for English high front [iy], low front [ae] and high back [uw]. 

**Front vowel** 

In the vowel [iy], for example, the highest point of the tongue is toward the front of the mouth. In the vowel [uw], by contrast, the high-point of the tongue is located toward the back of the mouth. Vowels in which the tongue is raised toward the front are called **front vowels** ; those in which the tongue is raised toward the 

**6** CHAPTER 25 • PHONETICS 

> **back vowel** back are called **back vowels** . Note that while both [ih] and [eh] are front vowels, the tongue is higher for [ih] than for [eh]. Vowels in which the highest point of the 

> **high vowel** tongue is comparatively high are called **high vowels** ; vowels with mid or low values of maximum tongue height are called **mid vowels** or **low vowels** , respectively. 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0006-02.png)


**----- Start of picture text -----**<br>
high<br>iy y uw uw<br>ih<br>uh<br>ey oy ax<br>front back<br>eh<br>ao<br>ah<br>ae aa<br>low<br>ay<br>aw<br>ow<br>**----- End of picture text -----**<br>


**Figure 25.5** The schematic “vowel space” for English vowels. 

**diphthong** 

**rounded vowel** 

Figure 25.5 shows a schematic characterization of the height of different vowels. It is schematic because the abstract property **height** correlates only roughly with actual tongue positions; it is, in fact, a more accurate reflection of acoustic facts. Note that the chart has two kinds of vowels: those in which tongue height is represented as a point and those in which it is represented as a path. A vowel in which the tongue position changes markedly during the production of the vowel is a **diphthong** . English is particularly rich in diphthongs. 

The second important articulatory dimension for vowels is the shape of the lips. Certain vowels are pronounced with the lips rounded (the same lip shape used for whistling). These **rounded** vowels include [uw], [ao], and [ow]. 

## **Syllables** 

**syllable** 

**nucleus** 

**onset coda rime** 

**syllabification phonotactics** 

Consonants and vowels combine to make a **syllable** . A syllable is a vowel-like (or **sonorant** ) sound together with some of the surrounding consonants that are most closely associated with it. The word _dog_ has one syllable, [d aa g] (in our dialect); the word _catnip_ has two syllables, [k ae t] and [n ih p]. We call the vowel at the core of a syllable the **nucleus** . Initial consonants, if any, are called the **onset** . Onsets with more than one consonant (as in _strike_ [s t r ay k]), are called **complex onsets** . The **coda** is the optional consonant or sequence of consonants following the nucleus. Thus [d] is the onset of _dog_ , and [g] is the coda. The **rime** , or **rhyme** , is the nucleus plus coda. Figure 25.6 shows some sample syllable structures. 

The task of automatically breaking up a word into syllables is called **syllabification** . Syllable structure is also closely related to the **phonotactics** of a language. The term **phonotactics** means the constraints on which phones can follow each other in a language. For example, English has strong constraints on what kinds of consonants can appear together in an onset; the sequence [zdr], for example, cannot be a legal English syllable onset. Phonotactics can be represented by a language model or finite-state model of phone sequences. 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0007-00.png)


**----- Start of picture text -----**<br>
25.3 • PROSODY 7<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0007-01.png)


**----- Start of picture text -----**<br>
σ σ σ<br>Onset Rime Onset Rime Rime<br>h Nucleus Coda g r Nucleus Coda Nucleus Coda<br>ae m iy n eh g z<br>**----- End of picture text -----**<br>


**Figure 25.6** Syllable structure of _ham_ , _green_ , _eggs_ . _σ_ =syllable. 

## 25.3 Prosody 

> **prosody Prosody** is the study of the intonational and rhythmic aspects of language, and in particular the use of **F0** , **energy** , and **duration** to convey pragmatic, affective, or conversation-interactional meanings.[1] Prosody can be used to mark **discourse structure** , like the difference between statements and questions, or the way that a conversation is structured. Prosody is used to mark the **saliency** of a particular word or phrase. Prosody is heavily used for paralinguistic functions like conveying affective meanings like happiness, surprise, or anger. And prosody plays an important role in managing turn-taking in conversation. 

## **25.3.1 Prosodic Prominence: Accent, Stress and Schwa** 

> **prominence** In a natural utterance of American English, some words sound more **prominent** than others, and certain syllables in these words are also more **prominent** than others. What we mean by prominence is that these words or syllables are perceptually more salient to the listener. Speakers make a word or syllable more salient in English by saying it louder, saying it slower (so it has a longer duration), or by varying F0 during the word, making it higher or more variable. 

> **pitch accent Accent** We represent prominence via a linguistic marker called **pitch accent** . Words or syllables that are prominent are said to **bear** (be associated with) a pitch accent. Thus this utterance might be pronounced by **accenting** the underlined words: 

(25.1) I’m a little surprised to hear it characterized as happy. 

**Lexical Stress** The syllables that bear pitch accent are called **accented** syllables. Not every syllable of a word can be accented: pitch accent has to be realized on the **lexical stress** syllable that has **lexical stress** . Lexical stress is a property of the word’s pronunciation in dictionaries; the syllable that has lexical stress is the one that will be louder or longer if the word is accented. For example, the word _surprised_ is stressed on its second syllable, not its first. (Try stressing the other syllable by saying SURprised; hopefully that sounds wrong to you). Thus, if the word _surprised_ receives a pitch accent in a sentence, it is the second syllable that will be stronger. The following example shows underlined accented words with the stressed syllable bearing the accent (the louder, longer syllable) in boldface: 

- (25.2) I’m a little sur **prised** to hear it **char** acterized as **hap** py. 

> 1 The word is used in a different but related way in poetry, to mean the study of verse metrical structure. 

**8** CHAPTER 25 • PHONETICS 

Stress is marked in dictionaries. The CMU dictionary (CMU, 1993), for example, marks vowels with 0 (unstressed) or 1 (stressed) as in entries for _counter_ : [K AW1 N T ER0], or _table_ : [T EY1 B AH0 L]. Difference in lexical stress can affect word meaning; the noun _content_ is pronounced [K AA1 N T EH0 N T], while the adjective is pronounced [K AA0 N T EH1 N T]. 

**reduced vowel schwa** 

**prominence** 

**Reduced Vowels and Schwa** Unstressed vowels can be weakened even further to **reduced vowels** , the most common of which is **schwa** ([ax]), as in the second vowel of _parakeet_ : [p ae r ax k iy t]. In a reduced vowel the articulatory gesture isn’t as complete as for a full vowel. Not all unstressed vowels are reduced; any vowel, and diphthongs in particular, can retain its full quality even in unstressed position. For example, the vowel [iy] can appear in stressed position as in the word _eat_ [iy t] or in unstressed position as in the word _carry_ [k ae r iy]. 

In summary, there is a continuum of prosodic **prominence** , for which it is often useful to represent levels like accented, stressed, full vowel, and reduced vowel. 

## **25.3.2 Prosodic Structure** 

Spoken sentences have prosodic structure: some words seem to group naturally together, while some words seem to have a noticeable break or disjuncture between **phrasingprosodic** them. Prosodic structure is often described in terms of **prosodic phrasing** , meaning that an utterance has a prosodic phrase structure in a similar way to it having a syntactic phrase structure. For example, the sentence _I wanted to go to London,_ **intonation phrase** _but could only get tickets for France_ seems to have two main **intonation phrases** , their boundary occurring at the comma. Furthermore, in the first phrase, there seems to be another set of lesser prosodic phrase boundaries (often called **intermediate intermediatephrasephrase phrase** s) that split up the words as _I wanted | to go | to London_ . These kinds of intonation phrases are often correlated with syntactic structure constituents (Price et al. 1991, Bennett and Elfner 2019). 

**intermediatephrasephrase** 

Automatically predicting prosodic boundaries can be important for tasks like TTS. Modern approaches use sequence models that take either raw text or text annotated with features like parse trees as input, and make a break/no-break decision at each word boundary. They can be trained on data labeled for prosodic structure like the Boston University Radio News Corpus (Ostendorf et al., 1995). 

## **25.3.3 Tune** 

**tune** 

**question rise** 

Two utterances with the same prominence and phrasing patterns can still differ prosodically by having different **tunes** . The **tune** of an utterance is the rise and fall of its F0 over time. A very obvious example of tune is the difference between statements and yes-no questions in English. The same words can be said with a final F0 rise to indicate a yes-no question (called a **question rise** ): 

You    know    what    I mean ? 

or a final drop in F0 (called a **final fall** ) to indicate a declarative intonation: 

You    know        what         I mean . 

Languages make wide use of tune to express meaning (Xu, 2005). In English, 

25.4 • ACOUSTIC PHONETICS AND SIGNALS **9** 

for example, besides this well-known rise for yes-no questions, a phrase containing a list of nouns separated by commas often has a short rise called a **continuation continuationrise rise** after each noun. Other examples include the characteristic English contours for expressing **contradiction** and expressing **surprise** . 

## **Linking Prominence and Tune** 

Pitch accents come in different varieties that are related to tune; high pitched accents, for example, have different functions than low pitched accents. There are many typologies of accent classes in different languages. One such typology is part of the **ToBI ToBI** (Tone and Break Indices) theory of intonation (Silverman et al. 1992). Each word in ToBI can be associated with one of five types of **pitch accents** shown in in Fig. 25.7. Each utterance in ToBI consists of a sequence of intonational phrases, **boundary tone** each of which ends in one of four **boundary tones** shown in Fig. 25.7, representing the utterance final aspects of tune. There are version of ToBI for many languages. 

||**Pitch Accents**||**Boundary Tones**|
|---|---|---|---|
|**H***|peak accent|**L-L%**|“fnal fall”: “declarative contour” of American|
||||English|
|**L***|low accent|**L-H%**|continuation rise|
|**L*+H**|scooped accent|**H-H%**|“question rise”:<br>cantonical yes-no question|
||||contour|
|**L+H***|rising peak accent|**H-L%**|fnal level plateau|
|**H+!H***|stepdown|||
|**Figure 25.7**<br>The accent and boundary tones labels from the ToBI transcription system for||||
|American|English intonation (Beckman and||Ayers 1997,Beckman and Hirschberg 1994).|



## 25.4 Acoustic Phonetics and Signals 

We begin with a very brief introduction to the acoustic waveform and its digitization and frequency analysis; the interested reader is encouraged to consult the references at the end of the chapter. 

## **25.4.1 Waves** 

Acoustic analysis is based on the sine and cosine functions. Figure 25.8 shows a plot of a sine wave, in particular the function 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0009-09.png)


where we have set the amplitude A to 1 and the frequency _f_ to 10 cycles per second. 

Recall from basic mathematics that two important characteristics of a wave are **frequency** its **frequency** and **amplitude** . The frequency is the number of times a second that a **amplitude** wave repeats itself, that is, the number of **cycles** . We usually measure frequency in **cycles per second** . The signal in Fig. 25.8 repeats itself 5 times in .5 seconds, hence **Hertz** 10 cycles per second. Cycles per second are usually called **hertz** (shortened to **Hz** ), so the frequency in Fig. 25.8 would be described as 10 Hz. The **amplitude** _A_ of a **period** sine wave is the maximum value on the Y axis. The **period** _T_ of the wave is the time it takes for one cycle to complete, defined as 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0009-12.png)


**10** CHAPTER 25 • PHONETICS 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0010-01.png)


**----- Start of picture text -----**<br>
1.0<br>0<br>–1.0<br>0 0.1 0.2 0.3 0.4 0.5<br>Time (s)<br>**----- End of picture text -----**<br>


**Figure 25.8** A sine wave with a frequency of 10 Hz and an amplitude of 1. 

Each cycle in Fig. 25.8 lasts a tenth of a second; hence _T_ = _._ 1 seconds. 

## **25.4.2 Speech Sound Waves** 

Let’s turn from hypothetical waves to sound waves. The input to a speech recognizer, like the input to the human ear, is a complex series of changes in air pressure. These changes in air pressure obviously originate with the speaker and are caused by the specific way that air passes through the glottis and out the oral or nasal cavities. We represent sound waves by plotting the change in air pressure over time. One metaphor which sometimes helps in understanding these graphs is that of a vertical plate blocking the air pressure waves (perhaps in a microphone in front of a speaker’s mouth, or the eardrum in a hearer’s ear). The graph measures the amount of **compression** or **rarefaction** (uncompression) of the air molecules at this plate. Figure 25.9 shows a short segment of a waveform taken from the Switchboard corpus of telephone speech of the vowel [iy] from someone saying “she just had a baby”. 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0010-06.png)


**----- Start of picture text -----**<br>
0.02283<br>0<br>–0.01697<br>0 0.03875<br>Time (s)<br>**----- End of picture text -----**<br>


**Figure 25.9** A waveform of the vowel [iy] from an utterance shown later in Fig. 25.13 on page 14. The _y_ -axis shows the level of air pressure above and below normal atmospheric pressure. The _x_ -axis shows time. Notice that the wave repeats regularly. 

The first step in digitizing a sound wave like Fig. 25.9 is to convert the analog representations (first air pressure and then analog electric signals in a microphone) **sampling** into a digital signal. This **analog-to-digital conversion** has two steps: **sampling** and **quantization** . To sample a signal, we measure its amplitude at a particular time; the **sampling rate** is the number of samples taken per second. To accurately measure a wave, we must have at least two samples in each cycle: one measuring the positive part of the wave and one measuring the negative part. More than two samples per cycle increases the amplitude accuracy, but fewer than two samples causes the frequency of the wave to be completely missed. Thus, the maximum frequency wave that can be measured is one whose frequency is half the sample rate (since every cycle needs two samples). This maximum frequency for a given sampling rate is **frequencyNyquistNyquist** called the **Nyquist frequency** . Most information in human speech is in frequencies below 10,000 Hz; thus, a 20,000 Hz sampling rate would be necessary for com- 

**frequencyNyquistNyquist** 

25.4 • ACOUSTIC PHONETICS AND SIGNALS **11** 

plete accuracy. But telephone speech is filtered by the switching network, and only frequencies less than 4,000 Hz are transmitted by telephones. Thus, an 8,000 Hz sampling rate is sufficient for **telephone-bandwidth** speech like the Switchboard corpus, while 16,000 Hz sampling is often used for microphone speech. 

**quantization** 

**channel** 

**PCM** 

Even an 8,000 Hz sampling rate requires 8000 amplitude measurements for each second of speech, so it is important to store amplitude measurements efficiently. They are usually stored as integers, either 8 bit (values from -128–127) or 16 bit (values from -32768–32767). This process of representing real-valued numbers as integers is called **quantization** because the difference between two integers acts as a minimum granularity (a quantum size) and all values that are closer together than this quantum size are represented identically. 

Once data is quantized, it is stored in various formats. One parameter of these formats is the sample rate and sample size discussed above; telephone speech is often sampled at 8 kHz and stored as 8-bit samples, and microphone data is often sampled at 16 kHz and stored as 16-bit samples. Another parameter is the number of **channels** . For stereo data or for two-party conversations, we can store both channels in the same file or we can store them in separate files. A final parameter is individual sample storage—linearly or compressed. One common compression format used for telephone speech is _µ_ -law (often written u-law but still pronounced mu-law). The intuition of log compression algorithms like _µ_ -law is that human hearing is more sensitive at small intensities than large ones; the log represents small values with more faithfulness at the expense of more error on large values. The linear (unlogged) values are generally referred to as **linear PCM** values (PCM stands for pulse code modulation, but never mind that). Here’s the equation for compressing a linear PCM sample value _x_ to 8-bit _µ_ -law, (where _µ_ =255 for 8 bits): 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0011-07.png)


There are a number of standard file formats for storing the resulting digitized wavefile, such as Microsoft’s .wav and Apple’s AIFF all of which have special headers; simple headerless “raw” files are also used. For example, the .wav format is a subset of Microsoft’s RIFF format for multimedia files; RIFF is a general format that can represent a series of nested chunks of data and control information. Figure 25.10 shows a simple .wav file with a single data chunk together with its format chunk. 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0011-09.png)


**Figure 25.10** Microsoft wavefile header format, assuming simple file with one chunk. Following this 44-byte header would be the data chunk. 

## **25.4.3 Frequency and Amplitude; Pitch and Loudness** 

Sound waves, like all waves, can be described in terms of frequency, amplitude, and the other characteristics that we introduced earlier for pure sine waves. In sound waves, these are not quite as simple to measure as they were for sine waves. Let’s consider frequency. Note in Fig. 25.9 that although not exactly a sine, the wave is 

**12** CHAPTER 25 • PHONETICS 

nonetheless periodic, repeating 10 times in the 38.75 milliseconds (.03875 seconds) captured in the figure. Thus, the frequency of this segment of the wave is 10/.03875 or 258 Hz. 

Where does this periodic 258 Hz wave come from? It comes from the speed of vibration of the vocal folds; since the waveform in Fig. 25.9 is from the vowel [iy], it is voiced. Recall that voicing is caused by regular openings and closing of the vocal folds. When the vocal folds are open, air is pushing up through the lungs, creating a region of high pressure. When the folds are closed, there is no pressure from the lungs. Thus, when the vocal folds are vibrating, we expect to see regular peaks in amplitude of the kind we see in Fig. 25.9, each major peak corresponding to an opening of the vocal folds. The frequency of the vocal fold vibration, or the **fundamentalfrequency** frequency of the complex wave, is called the **fundamental frequency** of the wave- **F0** form, often abbreviated **F0** . We can plot F0 over time in a **pitch track** . Figure 25.11 **pitch track** shows the pitch track of a short question, “Three o’clock?” represented below the waveform. Note the rise in F0 at the end of the question. 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0012-03.png)


**----- Start of picture text -----**<br>
500 Hz<br>0 Hz<br>three o’clock<br>0 0.544375<br>Time (s)<br>**----- End of picture text -----**<br>


**Figure 25.11** Pitch track of the question “Three o’clock?”, shown below the wavefile. Note the rise in F0 at the end of the question. Note the lack of pitch trace during the very quiet part (the “o’” of “o’clock”; automatic pitch tracking is based on counting the pulses in the voiced regions, and doesn’t work if there is no voicing (or insufficient sound). 

The vertical axis in Fig. 25.9 measures the amount of air pressure variation; pressure is force per unit area, measured in Pascals (Pa). A high value on the vertical axis (a high amplitude) indicates that there is more air pressure at that point in time, a zero value means there is normal (atmospheric) air pressure, and a negative value means there is lower than normal air pressure (rarefaction). 

In addition to this value of the amplitude at any point in time, we also often need to know the average amplitude over some time range, to give us some idea of how great the average displacement of air pressure is. But we can’t just take the average of the amplitude values over a range; the positive and negative values would (mostly) cancel out, leaving us with a number close to zero. Instead, we generally use the RMS (root-mean-square) amplitude, which squares each number before averaging (making it positive), and then takes the square root at the end. 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0012-07.png)


The **power** of the signal is related to the square of the amplitude. If the number 

**power** 

25.4 • ACOUSTIC PHONETICS AND SIGNALS **13** 

of samples of a sound is _N_ , the power is 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0013-02.png)


> **intensity** Rather than power, we more often refer to the **intensity** of the sound, which normalizes the power to the human auditory threshold and is measured in dB. If _P_ 0 is the auditory threshold pressure = 2 _×_ 10 _[−]_[5] Pa, then intensity is defined as follows: 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0013-04.png)


Figure 25.12 shows an intensity plot for the sentence “Is it a long movie?” from the CallHome corpus, again shown below the waveform plot. 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0013-06.png)


**----- Start of picture text -----**<br>
is it a long movie?<br>0 1.1675<br>Time (s)<br>**----- End of picture text -----**<br>


**Figure 25.12** Intensity plot for the sentence “Is it a long movie?”. Note the intensity peaks at each vowel and the especially high peak for the word _long_ . 

Two important perceptual properties, **pitch** and **loudness** , are related to fre- **pitch** quency and intensity. The **pitch** of a sound is the mental sensation, or perceptual correlate, of fundamental frequency; in general, if a sound has a higher fundamental frequency we perceive it as having a higher pitch. We say “in general” because the relationship is not linear, since human hearing has different acuities for different frequencies. Roughly speaking, human pitch perception is most accurate between 100 Hz and 1000 Hz and in this range pitch correlates linearly with frequency. Human hearing represents frequencies above 1000 Hz less accurately, and above this range, pitch correlates logarithmically with frequency. Logarithmic representation means that the differences between high frequencies are compressed and hence not as accurately perceived. There are various psychoacoustic models of pitch percep- **Mel** tion scales. One common model is the **mel** scale (Stevens et al. 1937, Stevens and Volkmann 1940). A mel is a unit of pitch defined such that pairs of sounds which are perceptually equidistant in pitch are separated by an equal number of mels. The mel frequency _m_ can be computed from the raw acoustic frequency as follows: 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0013-09.png)


As we’ll see in Chapter 26, the mel scale plays an important role in speech recognition. 

**14** CHAPTER 25 • PHONETICS 

The **loudness** of a sound is the perceptual correlate of the **power** . So sounds with higher amplitudes are perceived as louder, but again the relationship is not linear. First of all, as we mentioned above when we defined _µ_ -law compression, humans have greater resolution in the low-power range; the ear is more sensitive to small power differences. Second, it turns out that there is a complex relationship between power, frequency, and perceived loudness; sounds in certain frequency ranges are perceived as being louder than those in other frequency ranges. 

**pitch extraction** 

Various algorithms exist for automatically extracting F0. In a slight abuse of terminology, these are called **pitch extraction** algorithms. The autocorrelation method of pitch extraction, for example, correlates the signal with itself at various offsets. The offset that gives the highest correlation gives the period of the signal. There are various publicly available pitch extraction toolkits; for example, an augmented autocorrelation pitch tracker is provided with Praat (Boersma and Weenink, 2005). 

## **25.4.4 Interpretation of Phones from a Waveform** 

Much can be learned from a visual inspection of a waveform. For example, vowels are pretty easy to spot. Recall that vowels are voiced; another property of vowels is that they tend to be long and are relatively loud (as we can see in the intensity plot in Fig. 25.12). Length in time manifests itself directly on the x-axis, and loudness is related to (the square of) amplitude on the y-axis. We saw in the previous section that voicing is realized by regular peaks in amplitude of the kind we saw in Fig. 25.9, each major peak corresponding to an opening of the vocal folds. Figure 25.13 shows the waveform of the short sentence “she just had a baby”. We have labeled this waveform with word and phone labels. Notice that each of the six vowels in Fig. 25.13, [iy], [ax], [ae], [ax], [ey], [iy], all have regular amplitude peaks indicating voicing. 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0014-06.png)


**----- Start of picture text -----**<br>
she just had a baby<br>sh iy j ax s h ae dx ax b ey b iy<br>0 1.059<br>Time (s)<br>**----- End of picture text -----**<br>


**Figure 25.13** A waveform of the sentence “She just had a baby” from the Switchboard corpus (conversation 4325). The speaker is female, was 20 years old in 1991, which is approximately when the recording was made, and speaks the South Midlands dialect of American English. 

For a stop consonant, which consists of a closure followed by a release, we can often see a period of silence or near silence followed by a slight burst of amplitude. We can see this for both of the [b]’s in _baby_ in Fig. 25.13. 

Another phone that is often quite recognizable in a waveform is a fricative. Recall that fricatives, especially very strident fricatives like [sh], are made when a narrow channel for airflow causes noisy, turbulent air. The resulting hissy sounds have a noisy, irregular waveform. This can be seen somewhat in Fig. 25.13; it’s even clearer in Fig. 25.14, where we’ve magnified just the first word _she_ . 

25.4 • ACOUSTIC PHONETICS AND SIGNALS **15** 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0015-01.png)


**----- Start of picture text -----**<br>
she<br>sh iy<br>0 0.257<br>Time (s)<br>**----- End of picture text -----**<br>


**Figure 25.14** A more detailed view of the first word “she” extracted from the wavefile in Fig. 25.13. Notice the difference between the random noise of the fricative [sh] and the regular voicing of the vowel [iy]. 

## **25.4.5 Spectra and the Frequency Domain** 

While some broad phonetic features (such as energy, pitch, and the presence of voicing, stop closures, or fricatives) can be interpreted directly from the waveform, most computational applications such as speech recognition (as well as human auditory processing) are based on a different representation of the sound in terms of its component frequencies. The insight of **Fourier analysis** is that every complex wave can be represented as a sum of many sine waves of different frequencies. Consider the waveform in Fig. 25.15. This waveform was created (in Praat) by summing two sine waveforms, one of frequency 10 Hz and one of frequency 100 Hz. 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0015-05.png)


**----- Start of picture text -----**<br>
1<br>0<br>–1<br>0 0.5<br>Time (s)<br>**----- End of picture text -----**<br>


**Figure 25.15** A waveform that is the sum of two sine waveforms, one of frequency 10 Hz (note five repetitions in the half-second window) and one of frequency 100 Hz, both of amplitude 1. 

## **spectrum** 

We can represent these two component frequencies with a **spectrum** . The spectrum of a signal is a representation of each of its frequency components and their amplitudes. Figure 25.16 shows the spectrum of Fig. 25.15. Frequency in Hz is on the x-axis and amplitude on the y-axis. Note the two spikes in the figure, one at 10 Hz and one at 100 Hz. Thus, the spectrum is an alternative representation of the original waveform, and we use the spectrum as a tool to study the component frequencies of a sound wave at a particular time point. 

Let’s look now at the frequency components of a speech waveform. Figure 25.17 shows part of the waveform for the vowel [ae] of the word _had_ , cut out from the sentence shown in Fig. 25.13. 

Note that there is a complex wave that repeats about ten times in the figure; but there is also a smaller repeated wave that repeats four times for every larger pattern (notice the four small peaks inside each repeated wave). The complex wave has a frequency of about 234 Hz (we can figure this out since it repeats roughly 10 times 

**16** CHAPTER 25 • PHONETICS 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0016-01.png)


**----- Start of picture text -----**<br>
80<br>60<br>40<br>1 2 5 10 20 50 100 200<br>Frequency (Hz)<br>Hz)<br>/<br>Sound pressure level (dB<br>**----- End of picture text -----**<br>


**Figure 25.16** The spectrum of the waveform in Fig. 25.15. 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0016-03.png)


**----- Start of picture text -----**<br>
0.04968<br>0<br>–0.05554<br>0 0.04275<br>Time (s)<br>**----- End of picture text -----**<br>


**Figure 25.17** The waveform of part of the vowel [ae] from the word _had_ cut out from the waveform shown in Fig. 25.13. 

## in .0427 seconds, and 10 cycles/.0427 seconds = 234 Hz). 

The smaller wave then should have a frequency of roughly four times the frequency of the larger wave, or roughly 936 Hz. Then, if you look carefully, you can see two little waves on the peak of many of the 936 Hz waves. The frequency of this tiniest wave must be roughly twice that of the 936 Hz wave, hence 1872 Hz. 

Figure 25.18 shows a smoothed spectrum for the waveform in Fig. 25.17, computed with a discrete Fourier transform (DFT). 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0016-08.png)


**----- Start of picture text -----**<br>
20<br>0<br>–20<br>0 1000 2000 3000 4000<br>Frequency (Hz)<br>Hz)<br>/<br>Sound pressure level (dB<br>**----- End of picture text -----**<br>


**Figure 25.18** A spectrum for the vowel [ae] from the word _had_ in the waveform of _She just had a baby_ in Fig. 25.13. 

The _x_ -axis of a spectrum shows frequency, and the _y_ -axis shows some measure of the magnitude of each frequency component (in decibels (dB), a logarithmic measure of amplitude that we saw earlier). Thus, Fig. 25.18 shows significant frequency components at around 930 Hz, 1860 Hz, and 3020 Hz, along with many other lower-magnitude frequency components. These first two components are just what we noticed in the time domain by looking at the wave in Fig. 25.17! 

Why is a spectrum useful? It turns out that these spectral peaks that are easily visible in a spectrum are characteristic of different phones; phones have characteris- 

25.4 • ACOUSTIC PHONETICS AND SIGNALS **17** 

**cochlea** 

**spectrogram** 

tic spectral “signatures”. Just as chemical elements give off different wavelengths of light when they burn, allowing us to detect elements in stars by looking at the spectrum of the light, we can detect the characteristic signature of the different phones by looking at the spectrum of a waveform. This use of spectral information is essential to both human and machine speech recognition. In human audition, the function of the **cochlea** , or **inner ear** , is to compute a spectrum of the incoming waveform. Similarly, the acoustic features used in speech recognition are spectral representations. 

Let’s look at the spectrum of different vowels. Since some vowels change over time, we’ll use a different kind of plot called a **spectrogram** . While a spectrum shows the frequency components of a wave at one point in time, a **spectrogram** is a way of envisioning how the different frequencies that make up a waveform change over time. The _x_ -axis shows time, as it did for the waveform, but the _y_ -axis now shows frequencies in hertz. The darkness of a point on a spectrogram corresponds to the amplitude of the frequency component. Very dark points have high amplitude, light points have low amplitude. Thus, the spectrogram is a useful way of visualizing the three dimensions (time x frequency x amplitude). 

Figure 25.19 shows spectrograms of three American English vowels, [ih], [ae], and [ah]. Note that each vowel has a set of dark bars at various frequency bands, slightly different bands for each vowel. Each of these represents the same kind of spectral peak that we saw in Fig. 25.17. 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0017-06.png)


**----- Start of picture text -----**<br>
5000<br>0<br>0 2.81397<br>Time (s)<br>Frequency (Hz)<br>**----- End of picture text -----**<br>


**Figure 25.19** Spectrograms for three American English vowels, [ih], [ae], and [uh] 

**formant** 

Each dark bar (or spectral peak) is called a **formant** . As we discuss below, a formant is a frequency band that is particularly amplified by the vocal tract. Since different vowels are produced with the vocal tract in different positions, they will produce different kinds of amplifications or resonances. Let’s look at the first two formants, called F1 and F2. Note that F1, the dark bar closest to the bottom, is in a different position for the three vowels; it’s low for [ih] (centered at about 470 Hz) and somewhat higher for [ae] and [ah] (somewhere around 800 Hz). By contrast, F2, the second dark bar from the bottom, is highest for [ih], in the middle for [ae], and lowest for [ah]. 

We can see the same formants in running speech, although the reduction and coarticulation processes make them somewhat harder to see. Figure 25.20 shows the spectrogram of “she just had a baby”, whose waveform was shown in Fig. 25.13. F1 and F2 (and also F3) are pretty clear for the [ax] of _just_ , the [ae] of _had_ , and the [ey] of _baby_ . 

What specific clues can spectral representations give for phone identification? First, since different vowels have their formants at characteristic places, the spectrum can distinguish vowels from each other. We’ve seen that [ae] in the sample waveform had formants at 930 Hz, 1860 Hz, and 3020 Hz. Consider the vowel [iy] at the 

**18** CHAPTER 25 • PHONETICS 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0018-01.png)


**----- Start of picture text -----**<br>
she just had a baby<br>sh iy j ax s h ae dx ax b ey b iy<br>0 1.059<br>Time (s)<br>**----- End of picture text -----**<br>


**Figure 25.20** A spectrogram of the sentence “she just had a baby” whose waveform was shown in Fig. 25.13. We can think of a spectrogram as a collection of spectra (time slices), like Fig. 25.18 placed end to end. 

beginning of the utterance in Fig. 25.13. The spectrum for this vowel is shown in Fig. 25.21. The first formant of [iy] is 540 Hz, much lower than the first formant for [ae], and the second formant (2581 Hz) is much higher than the second formant for [ae]. If you look carefully, you can see these formants as dark bars in Fig. 25.20 just around 0.5 seconds. 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0018-04.png)


**----- Start of picture text -----**<br>
80<br>70<br>60<br>50<br>40<br>30<br>20<br>10<br>0<br>−10<br>0 1000 2000 3000<br>**----- End of picture text -----**<br>


**Figure 25.21** A smoothed (LPC) spectrum for the vowel [iy] at the start of _She just had a baby_ . Note that the first formant (540 Hz) is much lower than the first formant for [ae] shown in Fig. 25.18, and the second formant (2581 Hz) is much higher than the second formant for [ae]. 

The location of the first two formants (called F1 and F2) plays a large role in determining vowel identity, although the formants still differ from speaker to speaker. Higher formants tend to be caused more by general characteristics of a speaker’s vocal tract rather than by individual vowels. Formants also can be used to identify the nasal phones [n], [m], and [ng] and the liquids [l] and [r]. 

## **25.4.6 The Source-Filter Model** 

Why do different vowels have different spectral signatures? As we briefly mentioned above, the formants are caused by the resonant cavities of the mouth. The **sourcesource-filtermodel filter model** is a way of explaining the acoustics of a sound by modeling how the pulses produced by the glottis (the **source** ) are shaped by the vocal tract (the **filter** ). 

**harmonic** 

Let’s see how this works. Whenever we have a wave such as the vibration in air caused by the glottal pulse, the wave also has **harmonics** . A harmonic is another wave whose frequency is a multiple of the fundamental wave. Thus, for example, a 

25.5 • PHONETIC RESOURCES **19** 

115 Hz glottal fold vibration leads to harmonics (other waves) of 230 Hz, 345 Hz, 460 Hz, and so on on. In general, each of these waves will be weaker, that is, will have much less amplitude than the wave at the fundamental frequency. 

It turns out, however, that the vocal tract acts as a kind of filter or amplifier; indeed any cavity, such as a tube, causes waves of certain frequencies to be amplified and others to be damped. This amplification process is caused by the shape of the cavity; a given shape will cause sounds of a certain frequency to resonate and hence be amplified. Thus, by changing the shape of the cavity, we can cause different frequencies to be amplified. 

When we produce particular vowels, we are essentially changing the shape of the vocal tract cavity by placing the tongue and the other articulators in particular positions. The result is that different vowels cause different harmonics to be amplified. So a wave of the same fundamental frequency passed through different vocal tract positions will result in different harmonics being amplified. 

We can see the result of this amplification by looking at the relationship between the shape of the vocal tract and the corresponding spectrum. Figure 25.22 shows the vocal tract position for three vowels and a typical resulting spectrum. The formants are places in the spectrum where the vocal tract happens to amplify particular harmonic frequencies. 


![](raw/papers/2_LLM_NLP/NLP-自然语言处理/25-Phonetics_images/25-Phonetics.pdf-0019-05.png)


**----- Start of picture text -----**<br>
20<br>20<br>[iy]  (tea) 0 [uw]  (moo)<br>F1<br>[ae] (cat)<br>F2<br>F1 F2<br>0 0 –20 F1 F2<br>0 268 2416 4000 0 903 1695 4000 0 295 817 4000<br>Frequency (Hz) Frequency (Hz) Frequency (Hz)<br>[iy]  (tea) [ae] (cat)  [uw]  (moo)<br>Sound pressure level (dBHz)/ Sound pressure level (dBHz)/ Sound pressure level (dBHz)/<br>**----- End of picture text -----**<br>


**Figure 25.22** Visualizing the vocal tract position as a filter: the tongue positions for three English vowels and the resulting smoothed spectra showing F1 and F2. 

## 25.5 Phonetic Resources 

A wide variety of phonetic resources can be drawn on for computational work. On- **pronunciationdictionary** line **pronunciation dictionaries** give phonetic transcriptions for words. The LDC distributes pronunciation lexicons for Egyptian Arabic, Dutch, English, German, Japanese, Korean, Mandarin, and Spanish. For English, the CELEX dictionary (Baayen et al., 1995) has pronunciations for 160,595 wordforms, with syllabification, stress, and morphological and part-of-speech information. The open-source CMU Pronouncing Dictionary (CMU, 1993) has pronunciations for about 134,000 

**20** CHAPTER 25 • PHONETICS 

wordforms, while the fine-grained 110,000 word UNISYN dictionary (Fitt, 2002), freely available for research purposes, gives syllabifications, stress, and also pronunciations for dozens of dialects of English. 

Another useful resource is a **phonetically annotated corpus** , in which a collection of waveforms is hand-labeled with the corresponding string of phones. The **TIMIT corpus** (NIST, 1990), originally a joint project between Texas Instruments (TI), MIT, and SRI, is a corpus of 6300 read sentences, with 10 sentences each from 630 speakers. The 6300 sentences were drawn from a set of 2342 sentences, some selected to have particular dialect shibboleths, others to maximize phonetic diphone coverage. Each sentence in the corpus was phonetically hand-labeled, the sequence of phones was automatically aligned with the sentence wavefile, and then the automatic phone boundaries were manually hand-corrected (Seneff and Zue, 1988). **transcriptiontime-aligned** The result is a **time-aligned transcription** : a transcription in which each phone is associated with a start and end time in the waveform, like the example in Fig. 25.23. 

she had your dark suit in greasy wash water all year sh iy hv ae dcl jh axr dcl d aa r kcl s ux q en gcl g r iy s ix w aa sh q w aa dx axr q aa l y ix axr **Figure 25.23** Phonetic transcription from the TIMIT corpus, using special ARPAbet features for narrow transcription, such as the palatalization of [d] in _had_ , unreleased final stop in _dark_ , glottalization of final [t] in _suit_ to [q], and flap of [t] in _water_ . The TIMIT corpus also includes time-alignments (not shown). 

The Switchboard Transcription Project phonetically annotated corpus consists of 3.5 hours of sentences extracted from the Switchboard corpus (Greenberg et al., 1996), together with transcriptions time-aligned at the syllable level. Figure 25.24 shows an example . 

0.470 0.640 0.720 0.900 0.953 1.279 1.410 1.630 dh er k aa n ax v ih m b ix t w iy n r ay n aw **Figure 25.24** Phonetic transcription of the Switchboard phrase _they’re kind of in between right now_ . Note vowel reduction in _they’re_ and _of_ , coda deletion in _kind_ and _right_ , and resyllabification (the [v] of _of_ attaches as the onset of _in_ ). Time is given in number of seconds from the beginning of sentence to the start of each syllable. 

The Buckeye corpus (Pitt et al. 2007, Pitt et al. 2005) is a phonetically transcribed corpus of spontaneous American speech, containing about 300,000 words from 40 talkers. Phonetically transcribed corpora are also available for other languages, including the Kiel corpus of German and Mandarin corpora transcribed by the Chinese Academy of Social Sciences (Li et al., 2000). 

In addition to resources like dictionaries and corpora, there are many useful phonetic software tools. Many of the figures in this book were generated by the Praat package (Boersma and Weenink, 2005), which includes pitch, spectral, and formant analysis, as well as a scripting language. 

## 25.6 Summary 

This chapter has introduced many of the important concepts of phonetics and computational phonetics. 

- We can represent the pronunciation of words in terms of units called **phones** . The standard system for representing phones is the **International Phonetic** 

BIBLIOGRAPHICAL AND HISTORICAL NOTES **21** 

**Alphabet** or **IPA** . The most common computational system for transcription of English is the **ARPAbet** , which conveniently uses ASCII symbols. 

- Phones can be described by how they are produced **articulatorily** by the vocal organs; consonants are defined in terms of their **place** and **manner** of articulation and **voicing** ; vowels by their **height** , **backness** , and **roundness** . 

- Speech sounds can also be described **acoustically** . Sound waves can be described in terms of **frequency** , **amplitude** , or their perceptual correlates, **pitch** and **loudness** . 

- The **spectrum** of a sound describes its different frequency components. While some phonetic properties are recognizable from the waveform, both humans and machines rely on spectral analysis for phone detection. 

- A **spectrogram** is a plot of a spectrum over time. Vowels are described by characteristic harmonics called **formants** . 

## Bibliographical and Historical Notes 

The major insights of articulatory phonetics date to the linguists of 800–150 B.C. India. They invented the concepts of place and manner of articulation, worked out the glottal mechanism of voicing, and understood the concept of assimilation. European science did not catch up with the Indian phoneticians until over 2000 years later, in the late 19th century. The Greeks did have some rudimentary phonetic knowledge; by the time of Plato’s _Theaetetus_ and _Cratylus_ , for example, they distinguished vowels from consonants, and stop consonants from continuants. The Stoics developed the idea of the syllable and were aware of phonotactic constraints on possible words. An unknown Icelandic scholar of the 12th century exploited the concept of the phoneme and proposed a phonemic writing system for Icelandic, including diacritics for length and nasality. But his text remained unpublished until 1818 and even then was largely unknown outside Scandinavia (Robins, 1967). The modern era of phonetics is usually said to have begun with Sweet, who proposed what is essentially the phoneme in his _Handbook of Phonetics_ 1877. He also devised an alphabet for transcription and distinguished between _broad_ and _narrow_ transcription, proposing many ideas that were eventually incorporated into the IPA. Sweet was considered the best practicing phonetician of his time; he made the first scientific recordings of languages for phonetic purposes and advanced the state of the art of articulatory description. He was also infamously difficult to get along with, a trait that is well captured in Henry Higgins, the stage character that George Bernard Shaw modeled after him. The phoneme was first named by the Polish scholar Baudouin de Courtenay, who published his theories in 1894. 

Introductory phonetics textbooks include Ladefoged (1993) and Clark and Yallop (1995). Wells (1982) is the definitive three-volume source on dialects of English. 

Many of the classic insights in acoustic phonetics had been developed by the late 1950s or early 1960s; just a few highlights include techniques like the sound spectrograph (Koenig et al., 1946), theoretical insights like the working out of the source-filter theory and other issues in the mapping between articulation and acoustics ((Fant, 1960), Stevens et al. 1953, Stevens and House 1955, Heinz and Stevens 1961, Stevens and House 1961) the F1xF2 space of vowel formants (Peterson and Barney, 1952), the understanding of the phonetic nature of stress and the use of duration and intensity as cues (Fry, 1955), and a basic understanding of issues in phone perception (Miller and Nicely 1955,Liberman et al. 1952). Lehiste (1967) is 

**22** CHAPTER 25 • PHONETICS 

a collection of classic papers on acoustic phonetics. Many of the seminal papers of Gunnar Fant have been collected in Fant (2004). 

Excellent textbooks on acoustic phonetics include Johnson (2003) and Ladefoged (1996). Coleman (2005) includes an introduction to computational processing of acoustics and speech from a linguistic perspective. Stevens (1998) lays out an influential theory of speech sound production. There are a number of software packages for acoustic phonetic analysis. Probably the most widely used one is **Praat** (Boersma and Weenink, 2005). 

## Exercises 

- **25.1** Find the mistakes in the ARPAbet transcriptions of the following words: 

   - **a.** “three” [dh r i] **d.** “study” [s t uh d i] **g.** “slight” [s l iy t] 

   - **b.** “sing” [s ih n g] **e.** “though” [th ow] 

   - **c.** “eyes” [ay s] **f.** “planning” [p pl aa n ih ng] 

- **25.2** Ira Gershwin’s lyric for _Let’s Call the Whole Thing Off_ talks about two pronunciations (each) of the words “tomato”, “potato”, and “either”. Transcribe into the ARPAbet both pronunciations of each of these three words. 

- **25.3** Transcribe the following words in the ARPAbet: 

   1. dark 

   2. suit 

   3. greasy 

   4. wash 

   5. water 

- **25.4** Take a wavefile of your choice. Some examples are on the textbook website. Download the Praat software, and use it to transcribe the wavefiles at the word level and into ARPAbet phones, using Praat to help you play pieces of each wavefile and to look at the wavefile and the spectrogram. 

- **25.5** Record yourself saying five of the English vowels: [aa], [eh], [ae], [iy], [uw]. Find F1 and F2 for each of your vowels. 

Exercises 23 

Baayen, R. H., R. Piepenbrock, and L. Gulikers. 1995. _The CELEX Lexical Database (Release 2) [CD-ROM]_ . Linguistic Data Consortium, University of Pennsylvania [Distributor]. 

Beckman, M. E. and G. M. Ayers. 1997. Guidelines for ToBI labelling. Unpublished manuscript, Ohio State University, `http://www.ling.ohio-state. edu/research/phonetics/E_ToBI/` . 

- Beckman, M. E. and J. Hirschberg. 1994. The ToBI annotation conventions. Manuscript, Ohio State University. 

Bennett, R. and E. Elfner. 2019. The syntax–prosody interface. _Annual Review of Linguistics_ , 5:151–171. 

Boersma, P. and D. Weenink. 2005. Praat: doing phonetics by computer (version 4.3.14). [Computer program]. Retrieved May 26, 2005, from `http://www.praat.org/` . Clark, J. and C. Yallop. 1995. _An Introduction to Phonetics and Phonology_ , 2nd edition. Blackwell. 

- CMU. 1993. The Carnegie Mellon Pronouncing Dictionary v0.1. Carnegie Mellon University. 

Coleman, J. 2005. _Introducing Speech and Language Processing_ . Cambridge University Press. 

- Fant, G. M. 1960. _Acoustic Theory of Speech Production_ . Mouton. 

Fant, G. M. 2004. _Speech Acoustics and Phonetics_ . Kluwer. 

- Fitt, S. 2002. Unisyn lexicon. `http://www.cstr.ed.ac. uk/projects/unisyn/` . 

- Fry, D. B. 1955. Duration and intensity as physical correlates of linguistic stress. _JASA_ , 27:765–768. 

Greenberg, S., D. Ellis, and J. Hollenback. 1996. Insights into spoken language gleaned from phonetic transcription of the Switchboard corpus. _ICSLP_ . 

- Heinz, J. M. and K. N. Stevens. 1961. On the properties of voiceless fricative consonants. _JASA_ , 33:589–596. 

Johnson, K. 2003. _Acoustic and Auditory Phonetics_ , 2nd edition. Blackwell. 

- Koenig, W., H. K. Dunn, and L. Y. Lacy. 1946. The sound spectrograph. _JASA_ , 18:19–49. 

- Ladefoged, P. 1993. _A Course in Phonetics_ . Harcourt Brace Jovanovich. (3rd ed.). 

- Ladefoged, P. 1996. _Elements of Acoustic Phonetics_ , 2nd edition. University of Chicago. 

- Lehiste, I., editor. 1967. _Readings in Acoustic Phonetics_ . MIT Press. 

- Li, A., F. Zheng, W. Byrne, P. Fung, T. Kamm, L. Yi, Z. Song, U. Ruhi, V. Venkataramani, and X. Chen. 2000. CASS: A phonetically transcribed corpus of Mandarin spontaneous speech. _ICSLP_ . 

- Liberman, A. M., P. C. Delattre, and F. S. Cooper. 1952. The role of selected stimulus variables in the perception of the unvoiced stop consonants. _American Journal of Psychology_ , 65:497–516. 

Ostendorf, M., P. Price, and S. Shattuck-Hufnagel. 1995. The Boston University Radio News Corpus. Technical Report ECS-95-001, Boston University. 

Peterson, G. E. and H. L. Barney. 1952. Control methods used in a study of the vowels. _JASA_ , 24:175–184. 

Pitt, M. A., L. Dilley, K. Johnson, S. Kiesling, W. D. Raymond, E. Hume, and E. Fosler-Lussier. 2007. Buckeye corpus of conversational speech (2nd release). Department of Psychology, Ohio State University (Distributor). 

Pitt, M. A., K. Johnson, E. Hume, S. Kiesling, and W. D. Raymond. 2005. The buckeye corpus of conversational speech: Labeling conventions and a test of transcriber reliability. _Speech Communication_ , 45:90–95. 

Price, P. J., M. Ostendorf, S. Shattuck-Hufnagel, and C. Fong. 1991. The use of prosody in syntactic disambiguation. _JASA_ , 90(6). 

Robins, R. H. 1967. _A Short History of Linguistics_ . Indiana University Press, Bloomington. 

Seneff, S. and V. W. Zue. 1988. Transcription and alignment of the TIMIT database. _Proceedings of the Second Symposium on Advanced Man-Machine Interface through Spoken Language_ . 

Shoup, J. E. 1980. Phonological aspects of speech recognition. In W. A. Lea, editor, _Trends in Speech Recognition_ , pages 125–138. Prentice Hall. 

Silverman, K., M. E. Beckman, J. F. Pitrelli, M. Ostendorf, C. W. Wightman, P. J. Price, J. B. Pierrehumbert, and J. Hirschberg. 1992. ToBI: A standard for labelling English prosody. _ICSLP_ . Stevens, K. N. 1998. _Acoustic Phonetics_ . MIT Press. 

Stevens, K. N. and A. S. House. 1955. Development of a quantitative description of vowel articulation. _JASA_ , 27:484–493. 

Stevens, K. N. and A. S. House. 1961. An acoustical theory of vowel production and some of its implications. _Journal of Speech and Hearing Research_ , 4:303–320. 

Stevens, K. N., S. Kasowski, and G. M. Fant. 1953. An electrical analog of the vocal tract. _JASA_ , 25(4):734–742. 

Stevens, S. S. and J. Volkmann. 1940. The relation of pitch to frequency: A revised scale. _The American Journal of Psychology_ , 53(3):329–353. 

Stevens, S. S., J. Volkmann, and E. B. Newman. 1937. A scale for the measurement of the psychological magnitude pitch. _JASA_ , 8:185–190. 

Sweet, H. 1877. _A Handbook of Phonetics_ . Clarendon Press. Wells, J. C. 1982. _Accents of English_ . Cambridge University Press. 

Xu, Y. 2005. Speech melody as articulatorily implemented communicative functions. _Speech communication_ , 46(34):220–251. 

- Miller, G. A. and P. E. Nicely. 1955. An analysis of perceptual confusions among some English consonants. _JASA_ , 27:338–352. 

- NIST. 1990. TIMIT Acoustic-Phonetic Continuous Speech Corpus. National Institute of Standards and Technology Speech Disc 1-1.1. NIST Order No. PB91-505065. 

