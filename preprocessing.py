import pandas as pd
import csv


def data_preparation(file_name, dic_name, sl_name, split_file):
    # load sentences and splits
    df = pd.read_csv(file_name, delimiter='\t')
    df_split = pd.read_csv(split_file, delimiter=',')

    # clean sentences
    df['sentence'] = df['sentence'].apply(clean_sentence)

    # load dictionary
    dic = pd.read_csv(dic_name, delimiter='|', header=None)
    dic.columns = ['phrase', 'index']

    # load sentiment labels
    s_values = pd.read_csv(sl_name, delimiter='|')

    # generate a data frame consists of 'sentence' 'sentence id' 'label'
    df = get_sentiment_sentences_dataframe(df, dic, s_values)

    # split the data set
    df_train = df.loc[df_split['splitset_label'] == 1]
    df_test = df.loc[df_split['splitset_label'] == 2]
    df_dev = df.loc[df_split['splitset_label'] == 3]

    return df_train, df_dev, df_test


def get_sentimental_label(sentence, dic, s_values):
    u_count = 0

    index = dic['index'].loc[dic['phrase'] == sentence].values
    if not index:
        u_count += 1
        print(sentence)
        return 0.5, u_count
    index = index[0]
    label = s_values['sentiment values'].loc[s_values['phrase ids'] == index].values[0]
    return label, u_count


def get_sentiment_sentences_dataframe(df, dic, s_values):
    labels = [None] * df.shape[0]
    u_count = 0

    for index, row in df.iterrows():
        labels[index], u_count_temp = get_sentimental_label(row['sentence'], dic, s_values)
        u_count += u_count_temp

    print('Number of unmatched sentences: ' + str(u_count))

    df['score'] = labels
    df['fine_grained'] = pd.cut(df['score'],
                                [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                include_lowest=True,
                                labels=["very negative", "negative", "neutral", "positive", "very positive"])
    df['raw'] = pd.cut(df['score'],
                       [0, 0.5, 1.0],
                       include_lowest=True,
                       labels=["negative", "positive"])
    return df


def clean_sentence(sentence):
    replace_dict = {"-LRB-": "(",
                    "-RRB-": ")",
                    "Ã©": "é",
                    "Ã³": "ó",
                    "Ã­": "í",
                    "Ã¼": "ü",
                    "Ã¡": "á",
                    "Ã¦": "æ",
                    "Â": "",
                    "Ã ": "à",
                    "Ã¢": "â",
                    "Ã±": "ñ",
                    "Ã¯": "ï",
                    "Ã´": "ô",
                    "Ã¨": "è",
                    "Ã¶": "ö",
                    "Ã£": "ã",
                    "Ã»": "û",
                    "Ã§": "ç"}

    for key, value in replace_dict.items():
        sentence = sentence.replace(key, value)
    return sentence


if __name__ == "__main__":
    train, valid, test = data_preparation('data/datasetSentences.txt',
                                        'data/dictionary.txt',
                                        'data/sentiment_labels.txt',
                                        'data/datasetSplit.txt')

    train[['sentence']].to_csv('train.txt', header=None, index=None, sep='\t', quoting=csv.QUOTE_NONE)
    valid[['sentence']].to_csv('valid.txt', header=None, index=None, sep='\t', quoting=csv.QUOTE_NONE)
    test[['sentence']].to_csv('test.txt', header=None, index=None, sep='\t', quoting=csv.QUOTE_NONE)

    train[['score', 'fine_grained', 'raw']].to_csv('train_labels.csv', index=None)
    valid[['score', 'fine_grained', 'raw']].to_csv('valid_labels.csv', index=None)
    test[['score', 'fine_grained', 'raw']].to_csv('test_labels.csv', index=None)
