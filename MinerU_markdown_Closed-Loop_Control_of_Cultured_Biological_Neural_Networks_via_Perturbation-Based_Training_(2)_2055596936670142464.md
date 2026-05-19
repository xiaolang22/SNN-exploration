# Closed-Loop Control of Cultured Biological Neural Networks via Perturbation-Based Training

Hao Liang 

School of Electronic Engineering, 

Beijing University of Posts and 

Telecommunications 

State Key Laboratory of Information 

Photonics and Optical Communications 

Beijing, China 

liangsunsky@163.com 

Yin Deng 

School of Electronic Engineering, 

Beijing University of Posts and 

Telecommunications 

State Key Laboratory of Information 

Photonics and Optical Communications 

Beijing, China 

dengyin@bupt.edu.cn 

Shiyang Cao 

School of Electronic Engineering, 

Beijing University of Posts and 

Telecommunications 

State Key Laboratory of Information 

Photonics and Optical Communications 

Beijing, China 

shycao@bupt.edu.cn 

Zeying Lu 

School of Electronic Engineering, 

Beijing University of Posts and 

Telecommunications 

State Key Laboratory of Information 

Photonics and Optical Communications 

Beijing, China 

luzeying@bupt.edu.cn 

Yarong Lin 

Department of Biochemistry and Molecular 

Biology,Chinese 

Academy of Medical Sciences and Peking 

Union Medical College 

State Key Laboratory of Common 

Mechanism Research for Major Diseases 

Beijing, China 

linaong@163.com 

Longze Sha 

Department of Biochemistry and Molecular 

Biology,Chinese 

Academy of Medical Sciences and Peking 

Union Medical College 

State Key Laboratory of Common 

Mechanism Research for Major Diseases 

Beijing, China 

shalz_pumc@163.com 

Qi Xu 

Department of Biochemistry and Molecular 

Biology,Chinese 

Academy of Medical Sciences and Peking 

Union Medical College 

State Key Laboratory of Common 

Mechanism Research for Major Diseases 

Beijing, China 

xuqi@punc.edu.cn 

Lili Gui* 

School of Electronic Engineering, 

Beijing University of Posts and 

Telecommunications 

State Key Laboratory of Information 

Photonics and Optical Communications 

Beijing, China 

liligui@bupt.edu.cn 

Kun Xu 

School of Electronic Engineering, 

Beijing University of Posts and 

Telecommunications 

State Key Laboratory of Information 

Photonics and Optical Communications 

Beijing, China 

xukun@bupt.edu.cn 

Abstract—In this study, a closed-loop control system based on in vitro cultured biological neural networks (BNNs) is developed to validate their capabilities in nonlinear computation and adaptive learning. The BNNs are employed as a physical reservoir and trained using a combination of structured and perturbation stimuli. Neural spike signals are recorded via a multi-electrode array (MEA), and feature vectors are extracted for classification and behavioral decoding using the Bagging Tree algorithm. Experimental results show that the classification accuracy of the BNNs improves from $6 6 . 5 \%$ to $8 4 \%$ , achieving a kill rate exceeding $7 5 \%$ in a flight control task. These results demonstrate the plasticity and behavioral output capabilities of BNNs, offering a novel pathway for brain-inspired control systems. 

Keywords- biological neural networks; physical reservoir; perturbation stimuli; Bagging Tree algorithm 

# I. INTRODUCTION

With the development of brain-inspired intelligence and neuroengineering, in vitro cultured BNNs have emerged as novel information processing substrates, demonstrating significant potential in neural modeling, intelligent control, and biocomputation [1]. BNNs exhibit nonlinear computational capabilities [2] and synaptic plasticity [3], along with advantages 

such as high energy efficiency [4], strong parallelism, and adaptive behavior. In recent studies, BNNs have been frequently modeled as physical reservoir computing systems [5,6], utilizing their complex dynamic structures to achieve high-dimensional mapping and feature extraction, thus representing a promising approach for brain-inspired system implementation. 

The MEA is a core technology in in vitro BNN research [7], offering high spatiotemporal resolution and multi-channel interaction capabilities, and is widely used for neural stimulation and signal acquisition. In recent years, various studies have explored the integration of BNNs into practical application tasks. The Brainoware system combined brain organoids with MEAs to perform speech recognition and nonlinear equation prediction [8]. The DishBrain system established a neuron-driven table tennis control model, achieving short-term closed-loop interaction [9]. Other studies have trained BNNs using visual stimuli for pattern recognition tasks [10]. Although these efforts have demonstrated the feasibility of applying BNNs to specific tasks, limitations remain, including insufficient utilization of stimulation information, inadequate extraction of spatiotemporal features, and limited task complexity. As such, the full potential of BNNs in dynamic perception and adaptive learning has yet to be realized. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-18/27733971-90cc-4ba9-860e-d73a6cc53322/7565cde7a55a9cb60b9a231c7c7effaabe2ed0cf2864dd9ecd966da400c21b2c.jpg)



Figure 1. Intelligent control system schematic diagram. The system consists of four functional modules. First, the in vitro biological neural network module: the core computing unit of the system is the BNNs cultured on the MEA. Second, the MEA platform module realizes high-density signal acquisition and precise electrical stimulation of BNNs. Third, the recognition algorithm module: uses the Bagging Tree ensemble learning model to classify and identify the above feature vectors to generate instructions. Fourth, the airplane game module: realizes the encoding of the environment and the execution of instructions.


In this study, a BNN training method and encoding-decoding framework are proposed, integrating structured and perturbation-based electrical stimulation. A Bagging Tree algorithm is employed to decode multi-channel spiking patterns, enabling real-time control of an aircraft game. Experimental results show that, after training, the system achieves a hit rate exceeding $7 5 \%$ in the task, demonstrating the potential of BNNs in reservoir computing and biological intelligence applications. 

# II. EXPERIMENTAL SETUP AND METHOD

The experimental system, shown in Figure 1, consists of four core modules. The first is the in vitro BNNs module, which serves as the central computational unit and comprises BNNs cultured on an MEA and derived from cortical neurons of embryonic day 15.5 mice provided by Beijing Sibeifu Co., Ltd. These BNNs exhibit self-organized connectivity, nonlinear dynamics, and synaptic plasticity. The second module is the MEA platform, which facilitates high-density neural signal acquisition and precise electrical stimulation, acting as the interface between the neural network and the control task. Third, the recognition algorithm module employs a Bagging Tree ensemble learning model to classify spike-based feature vectors, mapping neural responses to discrete control commands. Finally, the aircraft game module encodes the relative positions of target aircraft and the controlled aircraft into electrical stimulation patterns delivered to the BNNs, while decoded outputs are used to drive the aircraft in a real-time closed-loop control system. 

As illustrated in Figure 2, the system encodes aircraft game information into stimulation inputs and decodes the corresponding responses of the BNNs. In Figure 2(a), the relative position between the enemy aircraft and the player's aircraft determines the stimulation region and amplitude. In Figure 2(b), within a post-stimulation time window of 100– 200 ms, the system acquires real-time spike data from the BNNs via the MEA platform. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-18/27733971-90cc-4ba9-860e-d73a6cc53322/12bde8c3d08624e548e741584ccfffd315a611710c18e473a9892c40fd53bc78.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-18/27733971-90cc-4ba9-860e-d73a6cc53322/2007f35041a467d40cf7c6297c3e50fe63cd76aa4ce957ce1b57b6548d6af202.jpg)



Figure 2. Encoding and Decoding Biological Neural Networks. (a) Encoding diagram. Encode the game information of the left and right positions and horizontal distances between the enemy and the aircraft as input information. (b) Decoding diagram. Sort the neural network activities and input them into the recognition algorithm model to generate instructions to control the aircraft.


Let the set of spikes collected within this time window be denoted as 

$$
S = \left\{c _ {1}, c _ {2}, \dots , c _ {n} \right\} \tag {1}
$$

where $c _ { i } \in \{ 1 , 2 , . . . , C \}$ denotes the electrode channel index of the $i - t h$ spike, $C$ is the total number of channels, and $n$ is the total number of spikes. The spike count for channel $j$ is defined as 

$$
N _ {j} = \sum_ {i = 1} ^ {n} I \left(c _ {i} = j\right) \tag {2}
$$

where $I ( \cdot )$ denotes the indicator function. Based on the values of $N _ { j }$ , the channels are sorted in descending order to construct an ordered sequence of active channel–spike count pairs 

$$
V = \operatorname {S o r t} _ {j} \left(N _ {j}\right), \quad j \in \{1, 2, \dots , C \} \tag {3}
$$

The indices of the top 100 most active channels are extracted to form a sorted vector $V _ { 1 0 0 }$ , which serves as the feature vector representing the neural response. If the number of active channels is less than 100, zero-padding is applied at the end to ensure consistent vector dimensionality. This feature vector is then fed into a pre-trained Bagging Tree classification model to identify the corresponding neural response category, which is subsequently translated into specific aircraft control commands, thereby completing the decoding process from neural spike activity to flight behavior. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-18/27733971-90cc-4ba9-860e-d73a6cc53322/44765a3c0e0c9d110311160a2b10a990a200acfef6bcf393c66b9c7a5c69d405.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-18/27733971-90cc-4ba9-860e-d73a6cc53322/2d1edf9ee5ac4da8ad06e4aec0d30d545495d635d1d43ec4e03b3f444d67725c.jpg)



Figure 3. Training biological neural networks. (a) Two regions and three stimulation amplitudes constitute six input modes. In the training method, green represents the input mode, while orange and red modes represent the rest interval. (b) Random stimulation effect diagram. The figure shows a 12-second grid diagram, with each dot representing the action potential of a neuron currently detected by the electrode.


Prior to executing control tasks, the BNNs must undergo electrical stimulation training, as illustrated in Figure 3. In Figure 3(a), overall spontaneous activity across the MEA chip is first scanned to identify two spatially distant and highly active regions, which are selected as input channels. Six stimulation patterns—biphasic pulses with amplitudes of $1 0 0 \mathrm { m V } _ { ; }$ , $3 0 0 \mathrm { m V }$ , and $5 0 0 \mathrm { m V }$ , each with a pulse width of $2 0 0 \mu \mathrm { s } .$ —are applied to these channels. Each pattern is repeated 20 times at 10-second intervals, forming one training round. A total of three training rounds are conducted. To enhance the network's pattern recognition capability, a perturbation stimulation mechanism is introduced within each training round. Specifically, between different stimulation patterns, 4 to 10 randomly timed biphasic pulses (amplitude: $2 0 0 { - } 6 0 0 \mathrm { m V }$ ; inter-pulse interval: 10 ms) are delivered. As shown in Figure 3(b), this perturbation design helps disrupt pre-existing network connectivity and reduces the frequency of synchronous network bursts (SNBs), thereby 

improving the memory capacity and response plasticity of the BNNs. 

# III.RESULTS AND DISCUSSION

The feature extraction method described in Figure 2(b) is applied to construct a pattern recognition model based on the Bagging Tree ensemble learning algorithm. In each training round, 120 feature vectors are extracted from 120 stimulation events and used as the training data for the Bagging Tree model. Given the limited size of the experimental dataset, ten-fold cross-validation is employed for model training and evaluation. To systematically assess the changes in BNN response characteristics and structural plasticity during training, spontaneous spiking activity of the network is recorded for 1 to 5 minutes before training and after each training round. These recordings are used to evaluate training-induced modifications in network structure. 

As shown in Figure 4(a), the changes in mean electrode correlation, modularity coefficient, and global efficiency across four stages—before training and after each of the three training rounds—illustrate the structural remodeling and functional plasticity of the BNNs during the training process. Initially, the network exhibits strong local synchrony, high modularity, and a loosely connected structure with evident redundancy in spike activity across channels. After the first round of training, a slight increase in correlation, a decrease in modularity, and an improvement in global efficiency are observed. These changes indicate a transition from local synchronization to structural differentiation, suggesting the onset of a decoupling– reorganization phase in the network. With continued training, correlation steadily increases, global efficiency further improves, and modularity significantly decreases. This progression reflects the formation of a more integrated and functionally effective connectivity pattern, enhancing the network’s overall information transmission efficiency and cooperative processing capability. 

As illustrated in Figure 4(b), the classification accuracy of the BNNs increased significantly from $6 6 . 5 \%$ before training to $8 4 \%$ after three rounds of training. This upward trend indicates that the network's ability to recognize and respond to external stimulation patterns was progressively enhanced during training. The spiking responses became more stable, and the extracted feature patterns more separable, reflecting a transition from spontaneous responses to task-driven responses. Combined with the structural analysis, the improvement in classification accuracy is found to be closely associated with the enhanced network integration, validating the mechanism of structure– function co-plasticity. 

It is worth emphasizing that the proposed training framework—based on the synergy of structured and perturbation stimulation—overcomes limitations of traditional approaches. By deeply integrating the spatiotemporal dynamics of BNNs for efficient feature encoding, and introducing perturbation to disrupt preexisting connectivity patterns, the network's plasticity is significantly enhanced. This mechanism enables BNNs to achieve superior performance in tasks of high complexity. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-18/27733971-90cc-4ba9-860e-d73a6cc53322/03ed6f31ceab7932f2c8f493d82bedcc3189ed25f6de6ae31089965aa2f701ae.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-18/27733971-90cc-4ba9-860e-d73a6cc53322/4fa8928f1181fe25790fa2a86d9470e129903528d011e95565f77f7c0ba6d17b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-18/27733971-90cc-4ba9-860e-d73a6cc53322/4553e478460d109ece9ad63d88a7767b5bf9a72209389c84da9829b57183514e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-18/27733971-90cc-4ba9-860e-d73a6cc53322/da444bac847a49d871638198092ac9c719845aa838a6e080ba7a8ef0c911b5fc.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-18/27733971-90cc-4ba9-860e-d73a6cc53322/900683e3ac543d88da09b1a761b410763c55142dbe3ce6e5681ac362c862539e.jpg)



Figure 4. Experimental results and data analysis diagram. (a) Average correlation coefficient, Modularity coefficient and Global efficiency change with the number of training rounds. The number of experimental samples $\scriptstyle \mathrm { n = 8 }$ , and the non-parametric significance test method (Wilcoxon rank-sum test) was used to perform statistical analysis on the network indicators between each stage. $\ast = \mathsf { p } < 0 . 0 5$ , $\ast \ast = \mathsf { p } < 0 . 0 1$ , and $\ast \ast \ast = \mathbf { p } < 0 . 0 0 1$ . (b) The confusion matrix represents the results of the model evaluation using ten-fold cross validation after training, which are $6 6 . 5 \%$ , $7 7 . 5 \%$ , and $8 4 \%$ , respectively.


# IV.CONCLUSION

This study integrates in vitro cultured BNNs with a real-time closed-loop control system to explore their potential in nonlinear computation and adaptive learning. By designing a training protocol that combines structured and perturbation-based electrical stimulation, the BNNs are utilized as a physical reservoir to extract spiking features, which are decoded using a Bagging Tree algorithm for aircraft control. Experimental results show that the classification accuracy of the BNNs improved from $6 6 . 5 \%$ to $8 4 \%$ over three training rounds, achieving an average hit rate of over $7 5 \%$ in a flight task, thereby demonstrating a clear transition from spontaneous to task-driven responses. Structural and functional analyses further reveal that training enhances global efficiency, reduces modularity, and promotes neuronal coordination and plasticity. These findings offer a verifiable and effective approach to the training and control of neural computing systems. 

# ACKNOWLEDGMENT

The authors acknowledge financial support from the National Natural Science Foundation of China (62401083), the Fundamental Research Funds for the Central Universities (ZDYY202102), the Fund of the State Key Laboratory of Information Photonics and Optical Communications (Beijing University of Posts and Telecommunications) (IPOC2021ZR02), and the CAMS Innovation Fund for Medical Sciences (2021- I2M-1-020). 

# REFERENCES



[1] Du, J., Deng, Y., Cao, S., Lu, Z., Li, J., Han, Z., ... & Xu, K. (2025). Biological intelligence computing based on in vitro neural networks: Key technologies and research status (Invited). Infrared and Laser Engineering, 54(3): 20250073–20250073. 





[2] Chen, Z., Liang, Q., Wei, Z., Chen, X., Shi, Q., Yu, Z., & Sun, T. (2023). An overview of in vitro biological neural networks for robot intelligence.Cyborg and Bionic Systems,4, 0001. 





[3] Isomura, T., Kotani, K., & Jimbo, Y. (2015). Cultured cortical neurons can perform blind source separation according to the free-energy principle. PLoS computational biology, 11(12), e1004643. 





[4] Bing, Z., Meschede, C., Röhrbein, F., Huang, K., & Knoll, A. C. (2018). A survey of robotics control based on learning-inspired spiking neural networks. Frontiers in neurorobotics, 12, 35. 





[5] Yada, Y., Yasuda, S., & Takahashi, H. (2021). Physical reservoir computing with force learning in a living neuronal culture.Applied Physics Letters,119(17). 





[6] Tanaka, G., Yamane, T., Héroux, J. B., Nakane, R., Kanazawa, N., Takeda, S., ... & Hirose, A. (2019). Recent advances in physical reservoir computing: A review. Neural Networks, 115, 100-123. 





[7] Spira, M. E., & Hai, A. (2013). Multi-electrode array technologies for neuroscience and cardiology. Nature nanotechnology, 8(2), 83-94. 





[8] Cai, H., Ao, Z., Tian, C., Wu, Z., Liu, H., Tchieu, J., ... & Guo, F. (2023). Brain organoid reservoir computing for artificial intelligence. Nature Electronics, 6(12), 1032-1039. 





[9] Kagan, B. J., Kitchen, A. C., Tran, N. T., Habibollahi, F., Khajehnejad, M., Parker, B. J., ... & Friston, K. J. (2022). In vitro neurons learn and exhibit sentience when embodied in a simulated game-world. Neuron, 110(23), 3952-3969. 





[10] Shao, W. W., Shao, Q., Xu, H. H., Qiao, G. J., Wang, R. X., Ma, Z. Y., ... & Li, X. H. (2025). Repetitive training enhances the pattern recognition capability of cultured neural networks. PLOS Computational Biology, 21(4), e1013043. 

