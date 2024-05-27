# [ACL 2024 Findings] MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning

<p align="center">
   📖 <a href="https://arxiv.org/abs/2311.10537" target="_blank">Paper</a>  
</p>


We propose a **Multi-disciplinary Collaboration (MC)** framework. The framework works in five stages: 
(i) expert gathering: gather experts from distinct disciplines according to the clinical question;
(ii) analysis proposition: domain experts put forward their own analysis with their expertise;
(iii) report summarization: compose a summarized report on the basis of a previous series of analyses;
(iv) collaborative consultation: engage the experts in discussions over the summarized report. The report will be revised iteratively until an agreement from all the experts is reached;
(v) decision making: derive a final decision from the unanimous report.

![](pics/overview.png)

## Requirements

Install all required python dependencies:

```
pip install -r requirements.txt
```

## Data

We evaluate our MC framework on two benchmark datasets MedQA, MedMCQA, and PubMedQA, as well as six subtasks most relevant to the medical domain from MMLU datasets including anatomy, clinical knowledge, college medicine, medical genetics, professional medicine, and college biology.

Please check our Google Drive: https://drive.google.com/file/d/11qNzDYIlimGGJ1fhQn2ux6w_rfFgJbyo/view?usp=sharing


## Implementations
Input your own openai api key in api_utils.py.
```
sh inference.sh
```

## Cite Us
If you find this project useful, please cite the following paper:

```
@article{tang2023MedAgents,
      title={ML-MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning}, 
      author={Xiangru Tang and Anni Zou and Zhuosheng Zhang and Yilun Zhao and Xingyao Zhang and Arman Cohan and Mark Gerstein},
      year={2023},
      journal={arXiv preprint arXiv:2311.10537},
}
```

