from pandas import read_csv
from json import dumps

from matplotlib.pyplot import plot, show

from src.features import get_features

train = read_csv('data/train_essays.csv')
prompts = read_csv('data/train_prompts.csv')

information_densities,quote_densities,punctuation_densities,novelties = [],[],[],[]
for sample_index in range(100):
    __train_sample = train.loc[sample_index]
    __user_id = __train_sample['id']
    is_generated = bool(__train_sample['generated'])
    __prompt_id = __train_sample['prompt_id']
    __prompt_data = prompts[prompts['prompt_id']==__prompt_id]
    #topic = __prompt_data['prompt_name'].values[0]
    #question = __prompt_data['instructions'].values[0]
    sources = __prompt_data['source_text'].values[0]
    answer = __train_sample['text']
    features = get_features(answer=answer, source_text=sources)
    information_densities.append(features['information_density'])
    quote_densities.append(features['quote_density_score'])
    punctuation_densities.append(features['punctuation_density_score'])
    novelties.append(features['novelty_score'])
    print(__user_id, is_generated, dumps(features,indent=2))

plot(information_densities)
plot(quote_densities)
plot(punctuation_densities)
plot(novelties)
show()