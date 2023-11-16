# MedAgents
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

## Implementations
Input your own openai api key in api_utils.py.
```
sh inference.sh
```
