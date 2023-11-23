![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F57034e38-1e43-43c0-9399-31d3f1f67620%2FUntitled.png?table=block&id=ed1c95ea-f7b6-47c1-b587-6d349feccac6&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)  
  
**트랜스포머(Transformer)** 는 2017년 구글이 발표하여 NIPS에 등재된 논문인 **“Attention is all you need”** 에서 나온 모델로 기존의 seq2seq의 구조인 인코더-디코더를 따르면서도, 논문의 타이틀처럼 어텐션(Attention)만으로 구현한 모델이다. 이 모델은 *RNN을 사용하지 않고, 인코더-디코더 구조를 설계하였음에도 번역 성능에서 RNN보다 우수한 성능을 보여주었다.*

### 🪧 ***Road Map***
——————————————————————————
**1.** Overview
**2.** Positional Encoding
**3.** Self-Attention (+ Multi-Head Attention)
**4.** Residual Learning
**5.** Add + Norm
**6.** Attention in Encoder and Decoder
**7.** Position-wise Feedforward Networks
**8.** Output Probabilities
**9.** Transformer : Attention Is All New Need


# Ⅰ. Overview

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F90636fb7-bfd7-49eb-a357-4109eb80277f%2FUntitled.png?table=block&id=81eeea3a-ec7c-4bb2-bef7-f0128271537e&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

**트랜스포머** 는 RNN을 사용하지 않지만 기존의 Seq2Seq처럼 인코더에서 입력 시퀀스를 입력받고, 디코더에서 출력 시퀀스를 출력하는 인코더-디코더 구조를 유지하고 있다. 크게 인코더에서 디코더로 흐름이 이어지며, 데이터가 입력되고 출력값이 나오는 흐름에 따라 순차적으로 개념을 알아보고 마지막에 전체정리를 하는 방식으로 설명하겠다.

![FIG 00. Transformer의 구조](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fbe5e61c6-ea1a-4f1d-994c-877de7f20b8f%2FUntitled.png?table=block&id=2463764b-c400-460d-b4f6-eba7b7d609eb&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

위와 같은 **Transformer**의 구조를 보면 알겠지만 이전까지 배웠던 Seq2Seq나 Attention Mechanism에서는 볼 수 없었던 생소한 개념이 많이 등장한다. 각 블록의 표기나 논문을 참조하면서 보면 아래와 같은 내용이 등장할 것이다.

- Positional Encoding
- Self-Attention
- Multi-Head Attention
- Residual Learning & Residual Connection
- Feed Forward
- Encoder & Decoder

이러한 각 개념을 먼저 알아본뒤 전체적인 **Transformer**의 흐름을 살펴보자. 

# Ⅱ. Positional Encoding

트랜스포머는 기존의 방식과는 패러다임이 다르다.

Attention Mechanism을 사용하지만 기초적인 모델인 RNN과 CNN이 전혀 사용되지 않았기때문에 **임베딩된 입력값의 위치를 알 수 없다.** **즉, 문장 내 각각의 단어에 대한 순서정보를 주기 위해 Positional Encoding이라는 기법을 사용**하여 전달해준다. 향후 BERT와 같은 모델에서도 채택한 중요한 개념이다.

자세히 알아보자.

## (1). Traditional Embedding

우리가 어떤 단어정보를 네트워크에 넣기 위해서는 일반적으로 임베딩 과정을 거친다.

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F9bd5d08c-4df5-4746-a75e-3d6f841bda54%2FUntitled.png?table=block&id=8b1556bf-1460-45af-9c57-cfeaecef9b45&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

맨 처음 입력 차원은 특정 언어에서 존재할 수 있는 단어의 개수와 같고 동시에 각각의 정보들은 원핫 인코딩 형태로 표현이 되기 때문에 일반적으로 네트워크에 넣을 때 **Embedding**을 거쳐 보다 적은 차원의 실수값으로 표현하여 넣는다. 

## (2). Positional Encoding

만약 **Transformer**가 기존의 모델들과 같이 RNN기반의 모델을 활용했었더라면 이 모델을 적용하는 것 만으로도 ***각각의 단어가 RNN에 들어갈 때 순서에 맞게 입력되어 각 셀의 히든스테이트는 자동적으로 순서에 대한 정보를 가지게 된다.*** 

다만, **Transformer**와 같이 RNN기반의 구조를 사용하지 않는다면 **특정한 단어가 어떠한 단위 앞(혹은 뒤)에 위치하는지 위치(=순서)정보를 포함한 임베딩을 사용해야 하고** 이를 위해  **Postional Encoding**을 사용한다.

![Fig-01. Encoder Input Embedding Parts : ‘Transformer : Attention is All You Need’, NIPS, 2017](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F0d2bcbe2-21e2-41ac-9bf5-ae8e54a71043%2FUntitled.png?table=block&id=c42670c2-d252-4560-8c1e-3cfcf250ef7e&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

**Input Embedding** 값과 **Positional Encoding**된 값을 ***각각 Element Wise로 더해*** 각 단어에 대한 위치정보를 네트워크가 알 수 있도록 하는 것이다. 이렇게 나온 값을 Attention에 넣어주게 되는 것이다. 
조금 더 깊게 알아보자.

### 1. Input Embedding

**Input Embedding**은 Input에 입력된 데이터를 컴퓨터가 이해할 수 있는 형태로 바꾸는 작업이다. 
우리가 잘 알고 있듯 Inputs으로 들어온 Corpus가 Integer Encoding을 한 뒤 그 값들을 가져가게 된다.

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fb6647f04-14fe-47f2-9df7-d92120c961e2%2FUntitled.png?table=block&id=707fa060-8b46-425f-a7d6-a9290b58f3db&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

가령, “This is my car”라는 문장이 주어졌을 때, 문장을 구성하는 각각의 단어는 그에 상응하는 인덱스 값에 매칭이 되고, 이 값들은 Input Embedding에 전달되는 것이다.

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fa2228366-64fa-4fb8-a3f7-44dd6d63d395%2FUntitled.png?table=block&id=3d924d0f-c6a0-43d6-9836-ecacb1972885&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

이때 각각의 단어 인덱스들은 저마다 다른 벡터값을 지니고 있는데 (논문에서는 이를 $d_{model}$이라고 하고 512를 사용했다.) 각각의 벡터 차원은 해당 단어의 정보를 가지며 서로 다른 단어의 정보가 유사할 수록 임베딩된 벡터공간에서의 거리가 가까울 것이다. (즉, 벡터공간에서의 거리가 가까운 두 단어가 유사함을 의미한다.)

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F3cd1fbb7-5b23-496f-97cb-246ae73c440f%2FUntitled.png?table=block&id=00489cc7-b4e3-488b-8f86-44aafbd15076&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

Embedding Layer는 Input index값들을 받아서 각각의 단어 임베딩 벡터값으로 바꿔주고 이렇게 나온 값들에 Positional Encoding의 벡터값을 더하는 연산을 하게 된다.

### 2. 단어의 위치정보

**Transformer** 구조에선 단어의 위치정보를 임베딩된 벡터값에 element wise로 더해준다고 했다. 
이에 대해 알아보기전 **위치정보는 왜 중요한지**에 대해 알아보자. 아래 2문장을 보자.

- Although I did not get 95 in last TOEFL, I could get in the Ph.D program.
- Although I did get 95 in last TOEFL, I could not get in the Ph.D program.

위 2개문장을 해서해보면, 
1번 문장은 “지난 토플시험에서 95점을 못받았지만, 박사과정에 입학할 수 있었다.”이고, 
2번 문장은 “지난 토플시험에서 95점을 받았지만, 박사과정에 입학하지 못했다.”로 해석이 된다. 
***즉, not의 위치에 따라 문장의 뜻이 완전히 달라지게 되는 것이다.***
그래서 임베딩 된 벡터값에 단어들의 위치정보를 더해줘야 하는데, 이때 지켜야 할 규칙이 2가지가 있다.

1. **모든 위치값은 시퀀스의 길이나 Input에 관계없이 동일한 식별자를 가져야 한다.** 
즉, 각 위치에 따른 단어가 바뀌더라도 위치 임베딩은 동일하게 유지될 수 있어야 한다.
    
    ![단어의 벡터값이 변해도 위치 임베딩 값은 변함이 없다!](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Ff97eee41-4574-48c5-822d-9d3b03b79379%2FUntitled.png?table=block&id=14d5e143-76e6-435e-89ac-b1a8aea01d53&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)
    
    단어의 벡터값이 변해도 위치 임베딩 값은 변함이 없다!
    
2. **모든 위치 임베딩값은 너무 크면 안된다.** 
벡터공간속 임베딩된 단어들이 위치값을 더해서 순서를 알게 되는데, 이때 위치 임베딩값이 너무 커져버리면 단어 간의 상관관계 및 의미를 유추할 수 있는 의미정보 값이 상대적으로 작아지게 되고, Attention Layer를 통과할 때 제대로 학습이 안될 수도 있다.
    
    ![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F29ee43a5-ee20-4fc5-9446-252787e81a09%2FUntitled.png?table=block&id=c9a05931-7dd7-4137-ad97-e152d5f6f09f&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)
    

### 3. 위치 벡터를 얻는 방법

위치 벡터를 부여하는 방법으로는 **‘주기함수’**를 사용한다.

앞서 말했던 위치 임베딩의 값이 단어에 관계없이 동일한 식별자를 가져야한다는 점과 그 값의 크기가 너무 크면 안되는 점을 고려했을 때 주기함수인 사인/코사인 함수가 적절하다는 것이다.

Sine & Cosine 함수는 -1과 1사이를 반복하는 주기함수로 1을 초과하지 않고 -1미만으로 떨어지지 않아 값의 범위가 너무 커지지 않는 조건을 만족한다.

**같은 위치의 단어(=토큰)는 항상 같은 위치 벡터값을 가지고 있어야 하고**, 서로 다른 위치의 토큰은 위치 벡터값이 서로 달라야 한다. 여기서 Sine & Cosine 함수는 주기함수의 특징때문에 위치 벡터값이 겹칠 수 있다는 것이다.

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fbaccb1a2-51f1-4e17-b985-99acb86a3b65%2FUntitled.png?table=block&id=9be9b1f7-e374-4fec-801c-5a8cbfa01e34&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

위 그림만 보면 $p_0$와 $p_8$의 위치 벡터값이 동일하다. 우리가 헷갈리면 안되는 점이 Positional Encoding값은 벡터값으로 **차원을 지닌다**는 점이다.

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F2784a3f2-ccbb-4466-aa21-3ea316150611%2FUntitled.png?table=block&id=b37c9b95-b254-42a6-be74-71295cab211a&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

즉, 위치 벡터값이 같아지는 문제를 해결하기 위해 Sine과 Cosine함수를 동시에 사용한다. 
만약 그림과 같이 하나의 위치 벡터가 4개의 차원으로 표현되면 각 요소는 서로 다른 4개의 주기를 가지게 되므로 서로 겹치지 않는다.

그럼에도 각 차원의 벡터값들의 차이가 크지 않다면 서로 다른 단어 벡터 간의 위치 정보 차이가 미미할 것이다. 이 경우 주기함수의 Frequency를 이전 주기함수보다 크게 주면되고, 마지막 차원의 벡터값이 채워질 때까지 서로 다른 frequency를 가진 Sine & Cosine을 번갈아가며 계산하다보면 결과적으로 충분히 다른 Positional Encoding 값을 지니게 된다. 이를 수식으로 표현하면 아래와 같다.

※ $pos$는 position을 말하며, $i$는 차원을 의미한다.

$$
PE(pos, 2i) = sin(pos/10000^{2i/d_{model}})\\PE(pos, 2i+1) = cos(pos/10000^{2i/d_{model}})
$$

### 4. Input Embedding과 Positional Encoding 간의 연산

Seq2Seq with Attention을 보면 Attention Mechanism (정확히는 dot product attention)의 후반부 계산 부분에서 Concatenate를 적용한 것을 알 수 있다.  여기에서는 왜 Concatenate가 아닌 Summation을 사용한 것일까?

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F848ee365-af81-4ed1-88b0-88755af8ce78%2FUntitled.png?table=block&id=8cbd55d3-d101-407e-a0cf-df6e600e7811&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

위 그림이 summation 대신 concatenate를 사용한 경우이다.

**Concatenate**를 사용하면 단어 의미 정보를 포함하고 있는 단어 벡터 뒤에 위치 정보를 포함하는 Positional Embedding이 연결된다. 이 경우 단어의 의미 정보는 자체 차원공간을 가지게 되며, 위치 정보 역시 마찬가지이다. 즉, 직교성질(orthogonal)에 의해 둘은 서로 전혀 관계없는 공간에 있게 된다. 

    🔥 위 pragraph에 대한 내용을 이해하기 어려워서 자문을 구했으나 답변을 얻지못했다.. 
    시간을 들여 추가적인 레퍼런스를 찾아 알게된 내용을 정리한다.

    결론부터 말하면, Element Wise Summation(이하 Add라 지칭함)과 Concatenate(이하 Con이라 지칭함)는 본인이 구성한 신경망 구조와 하고자하는 테스크의 목적에 맞게 적절하게 선택해야 한다고 한다.  (비록, 메모리 사용관점에서는 Element Wise Summation이 Concatenate보다 더 좋지만 말이다!)

    하나의 예시를 첨부하겠다.
    A가 3000원, B가 2000원, C가 5000원을 가지고 있다고 하자.
    여기서 총합을 구할 때 Add와 Concatenate의 방식차이를 설명하면 다음과 같다.
    - Add : 전체 총합이 10000원이다.
    - Con : A가 3000원, B가 2000원, C가 5000원을 가져서 전체 총합이 10000원이다.

    만약 7000원짜리 물건을 살 때 Add와 Con 2가지 방식의 차이는 무엇인가?
    - Add : 전체 총합만을 알고있으므로 7000원짜리 물건을 살 수있는지 없는지 판단이 빠르다.
    - Con : 전체 총합 중 누가 얼마를 냈는지 알고있으므로 7000원짜리 물건을 사기위해 누가 얼마를 
                    내야할 지 판단이 빠르다.

    즉, 이를 벡터의 관점에서 바라보게되면 다음과 같이 정리할 수 있겠다.
    - Add : 더해지는 합계값이 하나의 묶음으로 보고 잔차로 인식한다.(ex, residual learning)
    - Con : 추출한 feature의 위치(순서)값을 그대로 보존하고자 한다. 두 가지 feature들이 밀접한 관련성을 가지지 않을 때 사용하는게 더 좋다.

    어떠한 방식을 적용해도 무방하지만, 적절한 상황에 따라 선택해야 하고, 
    Transformer가 등재된 시점에선 컴퓨팅 리소스가 충분하지 않아서 Add(=Element Wise Summation)을 선택한것으로 추측된다.

</aside>

**Concatenate**는 정보의 섞임을 방지해 혼란을 피할 수 있게 해주지만 메모리, 파라미터, 런타임 등의 cost문제가 발생한다. 이에 반해, **Summation**은 단어 의미정보와 위치 정보간의 균형이 잘 맞춰지는데 정보가 뒤섞이는 문제를 해결할 수는 없다. 즉, 모델이 매우 크고 GPU의 성능이 좋다면 (Cost 문제가 해결이 된다면) **Concatenate**를 사용해도 괜찮다!

**Transformer** 논문은 NIPS에 2017년 등록된 논문으로 현재와 다르게 그당시 컴퓨팅 리소스가 충분하지 않았을 것으로 생각된다. 그렇기에 **Summation**을 선택한 것 같다.

**결국, Postional Encoding을 적용함으로써 다량의 단어 벡터들을 병렬적으로 한번에 처리할 수 있게 되는 것이다.**

## (3). 코드 구현

추후 **Transformer** 구현을 하면서 자세히 살펴보도록 하고 지금은 원리에 초점을 맞춰 간단하게 구현해보자.

```python
# Positional Encoding
import math
import matplotlib.pyplot as plt

n = 4 # 단어(word)의 개수
dim = 8 # 임베딩(embedding) 차원 수

def get_angles(pos, i, dim):
    angles = 1/math.pow(10000,(2*(i//2))/dim)
    return pos * angles

def get_positional_encoding(pos, i, dim):
    if i%2 == 0:
        return math.sin(get_angles(pos, i, dim))
    return math.cos(get_angles(pos, i, dim))

result = [[0] * dim for _ in range(n)]

for i in range(n):
    for j in range(dim):
        result[i][j] = get_positional_encoding(i, j, dim)

plt.pcolormesh(result, cmap='Blues')
plt.show()
```

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fe103e9ed-6c95-4fcb-a8cd-36fa38f0d276%2FUntitled.png?table=block&id=3d91958c-3e68-481d-8992-65fe4d63a9c5&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

# Ⅲ. Self-Attention

기존에 어텐션 기법(Attention Mechanism)이 Machine Translatoion Task에서 적용될 때 번역할 대상이 되는 문장에서 주목해야 될 특정 단어에 더 높은 가중치를 부과하여 ‘집중’한다면 얻고자 하는 답을 빠르게 얻을 수 있음을 이전에 공부했었다. 그렇다면 **Self-Attention**은 무엇일까? Attention앞에 ‘Self-’ 가 붙은것만 봐도 알 수 있듯이 같은 문장 내에서 단어들간의 관계, 즉 연관성을 고려하여 어텐션을 계산하는 방법을 말한다. Transformer에서는 Self-Attention이 핵심이고 Encoder와 Decoder 2가지 구조 모두 사용된다.

## (1). Query, Key, Value

**Attention**의 목표는 Value를 통해 가중합을 계산하는 것(=이렇게 구한 값이 Attention Value)이고, 각 Value의 가중치는 주어진 Query와 Key가 얼마나 유사한가에 따라 결정된다. 각 요소들에 대한 의미를 알아보자.

- **Query (쿼리)**
    - 입력 시퀀스에서 관련된 부분을 찾으려고 하는 정보소스
- **Key (키)**
    - 관계의 연관도를 결정하기 위해 Query와 비교하게 되는 벡터
- **Value (밸류)**
    - 특정 key에 해당하는 입력 시퀀스의 정보로 가중치를 구하는데 사용되는 벡터

## (2). Multi-Head Attention

**Transformer**에서 어텐션은 어떻게 사용될까? 아래 그림을 잘 살펴보자.

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F04a48848-5145-43aa-bb12-3a592a39998e%2FUntitled.png?table=block&id=306b5446-871d-4613-aae6-2debff01a1a6&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

**Transformer**에는 **Multi-Head Attention**이라는 기법이 적용되었다.
즉, 그림과 같이 ***Self-Attention을 병렬적으로 h번 학습***시켰다는 것인데 이에 대해 깊게 알아보기 위해서는 Self-Attention에 대해 먼저 깊게 알아보아야 한다. 그전에 Multi-Head Attention의 구조를 보라. Q, K, V가 Linear Layer를 통과하고 있지 않은가? 이에 대해 잠깐 이야기 하겠다.

### (2)-1. Linear Layer

**Transformer**의 전체구조를 보면 Input Embedding된 벡터와 **Postional Encoding**된 값이 각각 Element Wise하게 더해지고, 이 값이 Multi-Head Attention 블록으로 들어옴을 알 수 있다.

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F861f29ee-4da0-465a-83a9-c872185b4627%2FUntitled.png?table=block&id=a371e983-0884-4c78-aae5-22e1ae755952&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

즉, **Multi-Head Attention**이 일어나기 전에 Linear Layer가 있으므로 Linear Layer에 앞서말한 입력값이 들어온다는 것이다. 다시말해 각각의 Linear Layer에는 동일한 Embedding Vector + Positional Encoding 값이 입력된다.

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F0adfe312-636a-4f1e-a093-2f7d3136e843%2FUntitled.png?table=block&id=38d61d62-1d69-43ae-a0f0-97d4b65406b1&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

여기서 **Linear Layer를 투과시키는 이유**를 2가지로 설명하자면

- Linear Layer가 입력을 출력으로 매핑하는 역할을 하고
- Linear Layer가 행렬이나 벡터의 차원을 바꾸는 역할을 하기 때문이다.

정리하면, **입력값으로 들어오는 Q, K, V 각각의 차원을 줄여서 병렬 연산에 적합한 구조를 만들고자 하는 것!**

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fffde5517-7c17-42ac-8c4d-e92ac5fc5c92%2FUntitled.png?table=block&id=7b35b081-1760-4bac-a169-a1492c624235&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

### (2)-2. Attention Score

Linear Layer를 통과한 Q, K, V는 **Scaled Dot-Product Attention** 블록을 통과하게 되는데 1개의 Scaled Dot-Product Attention 블록은 아래 그림과 같은 구조를 가진다.

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fb39b6680-1c47-451a-93b4-4241534f81ad%2FUntitled.png?table=block&id=e972aa28-209b-4672-a5a5-c53f89f1dcf7&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

여기부터는 우리가 아는 어텐션과 같다. Q와 K 행렬의 행렬곱(=MatMul, 행렬간의 유사도 의미)을 수행하는데 여기서 **Self-Attention**이 등장한다.

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F9fd08031-0f77-415f-b3b7-d31b828f71ca%2FUntitled.png?table=block&id=cf1bcb36-6d15-449e-9b67-8ae7fc05f58e&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

Self-Attention을 잠깐 짚고 넘어가자. 
이것도 Attention의 한 종류이기 때문에 쿼리(Query), 키(Key), 밸류(Value)의 3요소로 구성된다. 
다만 일반적인 Attention과 다른점이 있다면 한 문장 안에서 단어들 사이의 문맥적 관계성을 추출하는 과정이라는 점이다.

아래 수식처럼 입력 벡터 시퀀스($\mathbf{X}$)에 Query, Key, Value를 만들어주는 행렬($\mathbf{W}$)을 각각 곱한다. 

$$
\mathbf{Q} = \mathbf{X} \times \mathbf{W}_{Q}\\
\mathbf{K} = \mathbf{X} \times \mathbf{W}_{K}\\
\mathbf{V} = \mathbf{X} \times \mathbf{W}_{V}
$$

위와 같이 만든 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$값을 가지고 아래 셀프 어텐션 정의에 입각하여 계산한다.

$$
Attention(\mathbf{Q},\mathbf{K},\mathbf{V}) = softmax(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_K}})\mathbf{V}
$$

**쿼리와 키를 행렬곱** $(= \mathbf{Q}\mathbf{K}^T)$ 한 뒤 
해당 행렬의 **모든 요소값을 키 차원수의 제곱근 값으로 나눠** $(= \frac{1}{\sqrt{d_K}})$ 주고, 
이 행렬을 **행(row)단위로 소프트맥스(softmax)를 취해** $(= softmax(\;) )$ 
스코어 행렬을 만들어 준다. 
이렇게 만든 스코어 행렬에 **밸류를** $(= \mathbf{V})$ **행렬곱** 해주어 **Self-Attention** 계산을 마친다. 

이때 softmax 함수 안에 들어가는 수식 $\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_K}}$는 우리가 잘 아는 코사인 유사도 공식이다. 
모양이 좀 다르게 보일 수 있지만, 의미를 고려한다면 코사인 유사도와 별반 다르지 않음을 알 것이다.
*코사인 유사도는 두 벡터가 유사할 수록 값이 1에 가까워지고 서로 다를 수록 -1에 가까워지는 특징을 지닌다.* 

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F23a8f3a7-0b94-43ed-9456-fb81410b17fc%2FUntitled.png?table=block&id=fe0233b8-b987-4dbb-8079-3cd06e85f9fd&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

즉, 코사인 유사도는 벡터의 곱을 두 벡터의 L2 norm 곱, 즉 스케일링으로 나눈 값이다. 
이를통해 행렬의 유사도를 구할 수 있는데 우리가 보는 Attention의 경우 행렬 간의 곱으로 발생하는 차원 충돌을 피하기 위해 행렬 B를 전치행렬로 형태변환하여 곱해준다.

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F9748e8d0-04b2-4bd5-bde0-42ee2dfc398e%2FUntitled.png?table=block&id=1d779280-c0a6-4d26-b383-412d9d8eda4f&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

이렇게 구한 값은 **Attention Score**라고 하는데 가만히 살펴보면 자기 자신과 매핑되는 값이 가장 크고 그 다음으로 유사한 값이 크다는 것을 알 수 있다. 즉 Scaling이 필요하게 된것이다!

### (2)-3. Scaling & Softmax

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F0aab34dc-1b5d-482c-9e14-ba53833251cd%2FUntitled.png?table=block&id=a1b25ac7-e4a0-4640-99dd-90d0e6ad4bab&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

스케일링을 왜 하는걸까? dot-product 계산은 특성상 문장의 길이($d$)가 길어질 수록 더 큰 숫자를 가지게 된다. 문제는 나중에 softmax를 취했을 때 특정한 값만 과도하게 살아남고 나머지 값들은 완전히 죽어버리는 경우가 발생한다. 즉, **Scaling 연산을 수행하여 Gradient를 살려야 하는 것**이다. 그 후 Attention Score 행렬의 유사도를 0~1 사이 값으로 normalize하기 위해 Softmax를 사용한다. 

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fb86c448f-579e-442a-a015-ed3e677dcc98%2FUntitled.png?table=block&id=7108cd1c-7fb3-4350-b767-98e6546d8540&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

Attention Score와 앞서 구했던 Value를 내적하면 **Self-Attention Value**를 구하게 되면서 전체적인 Self-Attention이 마무리 된다.

### (2)-4. 코드 구현 (Attention Mechanism)

```python
import torch
import numpy as np
from torch.nn.functional import softmax

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device\n")
print(f"MPS 장치를 지원하도록 build가 되었는가? {torch.backends.mps.is_built()}")
print(f"MPS 장치가 사용 가능한가? {torch.backends.mps.is_available()}")

# ============================================================================================
# 1. 변수 정의
# ============================================================================================
x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 2.0, 0.0, 2.0],
    [1.0, 1.0, 1.0, 1.0],
])

w_query = torch.tensor([
    [1.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0]
])
w_key = torch.tensor([
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0]
])
w_value = torch.tensor([
    [0.0, 2.0, 0.0],
    [0.0, 3.0, 0.0],
    [1.0, 0.0, 3.0],
    [1.0, 1.0, 0.0]
])

# ============================================================================================
# 2. Q, K, V 만들기
# ============================================================================================
keys = torch.matmul(x, w_key)
querys = torch.matmul(x, w_query)
values = torch.matmul(x, w_value)

# ============================================================================================
# 3. Attention Score 만들기
# ============================================================================================
attn_scores = torch.matmul(querys, keys.T)
print(f"Attention Score : {attn_scores}")

# ============================================================================================
# 4. 각 Attention Score에 Sqrt(d_K)로 나눠 Softmax 취해주기
# ============================================================================================
key_dim_sqrt = np.sqrt(keys.shape[-1])
attn_scores_softmax = softmax(attn_scores / key_dim_sqrt, dim=1)
print(f"Attention Score with Softmax : {attn_scores_softmax}")

# ============================================================================================
# 5. Softmax를 취해 얻은 Attention Distribution과 Value Vector들을 가중합하여 Attention Value를 구하기
# ============================================================================================
attn_values = torch.matmul(attn_scores_softmax, values)
print(f"Attention Values : {attn_values}")

for idx, row in enumerate(attn_values):
    print(f"Max prob in {idx}th row : index is {np.argmax(row).item()}, value is {row[np.argmax(row).item()]}")
```

```python
Using mps device

MPS 장치를 지원하도록 build가 되었는가? True
MPS 장치가 사용 가능한가? True

Attention Score : tensor([[ 2.,  4.,  4.],
                                            [ 4., 16., 12.],
                                            [ 4., 12., 10.]])
Attention Score with Softmax : tensor([[1.3613e-01, 4.3194e-01, 4.3194e-01],
                                                                    [8.9045e-04, 9.0884e-01, 9.0267e-02],
                                                                        [7.4449e-03, 7.5471e-01, 2.3785e-01]])
Attention Values : tensor([[1.8639, 6.3194, 1.7042],
                                                [1.9991, 7.8141, 0.2735],
                                                [1.9926, 7.4796, 0.7359]])

Max prob in 0th row : index is 1, value is 6.319371223449707
Max prob in 1th row : index is 1, value is 7.814123153686523
Max prob in 2th row : index is 1, value is 7.479635715484619
```

### (2)-5. Multi-Head Attention

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F2e0f9059-0cbe-476a-92bd-a6099a6dd583%2FUntitled.png?table=block&id=6a4516ce-1dde-48dd-8d51-c2b2b3818342&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

**Transformer**는 앞서 설명한 Self-Attention을 병렬로 h번 학습시키는 **Multi-Head Attention** 구조로 이루어져 있다. Multi-Head를 사용한다는 것은 병렬적으로 학습을 진행하면서 여러 부분에 동시다발적으로 어텐션을 가할 수 있어 모델이 입력 토큰 간 다양한 유형의 종속성을 포착하고 동시에 다양한 소스의 정보를 결합할 수 있게 된다. 

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F6b9962f1-1ca3-4e89-9a58-35780dba5726%2FUntitled.png?table=block&id=f0fbddf3-8a61-4d4b-9275-142d68072bd0&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

즉, Multi-Head Attention을 사용하게 되면 각 head는 입력 시퀀스의 서로 다른 부분에 어텐션을 주기 때문에 모델이 입력 토큰간의 더 복잡한 관계를 다룰 수 있어 **하나의 문장이라 하더라도 각기 다른 종류의 어텐션이 모이게 되어 더 많은 정보로 표현이 가능**하다. 이는 다양한 유형의 종속성을 포착할 수 있고 더 정확한 답변을 내는데 도움이 되며, 표현력이 향상된다.

# Ⅲ. Residual Learning

## (1). Residual Connection

**Residual Connection**은 ResNet에서 등장한 개념이다. 과거 GoogleNet, VGG등으로 Deep CNN에 대한 연구가 진행되었으나 깊이가 깊어질수록 학습이 원활하게 잘 되지 않고 있음을 확인했었다. 다시말해 **너무 복잡하고 깊은 구조를 가진 네트워크는 오히려 성능이 좋지 않을수도 있다는 것**이다. 

기본적으로 Parameter 숫자가 많으면 많을수록 Overfitting이 잘 일어나게 된다. 이러한 문제를 해결하고자 한 CV쪽 모델이 바로 ResNet이라고 하고 이에 사용된 개념이 Residual Connection과 Residual Learning이다.

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Faf106072-b1d1-4a4c-a3e5-2640dc50e41a%2FUntitled.png?table=block&id=928fb109-2b72-47ea-959a-10f00afc7038&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

**Residual Connection**에 대해 잘 알아보기위해 잠시 ResNet 내용을 가볍게 살펴보자. ResNet 논문에서 제시된 Residual Connection이 앞서 얘기한 문제들을 해결할 수 있는데 여기서 Residual Connection이 위쪽 그림의 오른쪽 구조와 같이 마지막 활성화함수 $ReLU$를 거치기 전에 Input X를 더해주는 방식을 의미한다. 그에비해 기존 방식(왼쪽)의 경우 Input X에 대한 내용이 존재하지 않고 오직 $\hat y$를 통해서 학습이 진행되는 반면 ResNet 논문에서 제기된 구조(오른쪽)의 경우 $\hat y = x + F(x)$이다. 이를 반복적으로 Residual을 거치게 되면 아래와 같은 수식이 나타난다.

$$
x_{l+1} = x_1 + F(x_l)\\x_{l+2} = x_{l+1} + F(x_{l+1}) = x_l + F(x_l) + F(x_{l+1})
$$

위와 같은 구조가 반복되면서 결국 아래와 같은 식이 완성된다.

$$
x_L = x_l + \sum_{i=l}^{L-1}F(x_i)
$$

즉, **특정 위치의 출력은 특정 위치에서의 입력과 Residual 함수의 합으로 표현이 가능해 학습구조가 단순화되는 것**이다. 결과적으로 작은 잔차만을 학습하는 해당 방식을 **Residual Connection**이라 하는 것이다.

## (2). Residual Connections in Transformer

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Febf64c43-9296-4b91-9148-ed84d5fa40da%2FUntitled.png?table=block&id=d740ca58-cf8e-4529-96f7-23d863ff2e31&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

그렇다면 **Transformer**에서 Residual Learning이 어디에 사용될까?

위 그림에 보이는 빨간 점 위치마다 Residual Connection이 사용된 것을 확인할 수 있다. 앞서 말했듯 DNN구조가 깊어지고 복잡해지면 Parameter 숫자가 많아지고 이는 Overfitting이 될 확률을 늘린다. 한편 NLP분야의 task가 CV의 task보다 더 Gradient Vanishing/Exploding 되기 쉽다고 한다. Residual Connection은 이를 보완할 수 있는 메커니즘으로 Transformer에도 적극채용되었다. 

**Transformer**에서의 ***Residual Connection***은 크게 2가지 효과를 선보인다.

- **Gradient Vanishing / Exploding을 방지**하기 위해 사용된다.
- Attention을 진행하며 내부에 신경망을 투과시킴에 따라 가중치를 조정하기 위해 Back-Prop을 수행하게 되는데 이때 발생하는 Positional Encoding 벡터의 값이 희미해진다. **Positional Encoding 벡터를 손실없이 상위 레이어로 전달하기 위해 사용된다.**

# Ⅳ. Add + Norm

**Normalizatoin**은 정규화를 의미한다. 우리는 익숙하게 Batch Normalization은 들어봤어도 Layer Normalization은 익숙하지 않을 수 있다. 양쪽 다 살펴보자.

## (1). Batch Normalization

신경망에 각 Layer에 들어가는 input을 batch 단위의 평균과 분산으로 정규화해 학습을 효율적으로 만드는 방법이다. 이런 방식은 Neural Network의 각 층마다 입력값의 분포가 달라지는 현상을 없애기 위해 제안되었다. 그럼에도 단점이 존재하는데 이는 다음과 같다.

- Mini-batch의 크기에 의존적이다.
- Recurrent based model에 적용이 어렵다.
    - time-step의 개념이 적용되기 때문, 즉 매 time step마다 별도의 통계량이 적용되기 때문

우리는 시계열 데이터를 다뤄야하기에 Batch Normalization이 적합하지 않다는 것이다. 그래서 Layer Normalization을 도입하게 된다.

## (2). Layer Normalization

참고자료를 찾다가 발견한 Batch Normalization과 Layer Normalization의 차이를 논문적 인용을 이용해 적은것을 가져와 보겠다.

- Batch Normalization
    - Estimate the normalization statistics from the summed inputs to the neurons over a mini-batch of training case
- Layer Normalization
    - Estimate the normalization statistics from the summed inputs to the neurons within a hidden layer

직관적인 표현이라 생각한다. 즉, **Layer Normalization은 mini-batch 단위가 아니라 input을 기준으로 평균과 분산을 계산하게 된다.**

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fe6c52393-d0ad-4804-8ead-cff8303e808b%2FUntitled.png?table=block&id=5448ce48-ed1d-4986-8114-3a4627aa2569&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

# Ⅴ. Attention in Encoder and Decoder

Transformer에는 3가지 종류의 어텐션(Attention) 레이어가 사용된다.
사용되는 어텐션은 항상 Multi-Head Attention이고 사용되는 위치에 따라 다음과 같이 분류할 수 있다.

- **Encoder Self-Attention**
    - 각각의 단어가 서로에게 어떠한 연관성을 가지는지를 어텐션을 통해 구하도록 만들고, 전체 문장에 대한 표현방식(=Representation)을 학습 할 수 있게 만드는 것이 특징이다.
    
    ![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fc0b7e7f5-36c0-4d45-bccd-f2bd80f6df4d%2FUntitled.png?table=block&id=18dd947f-b790-46cb-b5e3-314b0fa2e477&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)
    
- **Masked Decoder Self-Attention**
    - 각각의 단어가 앞쪽에 출현한 단어들만을 참고하도록 만든다.
    
    ![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F748ea33e-82c5-48bc-9be5-8b1ddf63696f%2FUntitled.png?table=block&id=7f751141-21b6-47c6-9b95-bb25ddba7d68&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)
    
- **Encoder-Decoder Attention**
    - Query가 Decoder에 있고, Key와 Value가 Encoder에 있는 Attention 구조를 의미한다.
    
    ![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F7349a69a-cc07-4563-9a3d-116e17903322%2FUntitled.png?table=block&id=367970a0-445d-42e9-bc79-22107c6af649&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)
    

## (1). Masked Multi-Head Attention

Mask(마스크)란 무엇을 의미할까?

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F3b63e6b0-e505-4b4d-a4cd-d177f5546376%2FUntitled.png?table=block&id=a2107d69-0a32-492f-8953-5827de70c30d&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

Masked 혹은 Masking이라는 용어는 표현 그대로 무언가로 가린다는 의미를 가진다. 이 개념이 왜 등장했는지 부터 파악하도록 하자.

기존의 시계열 모델구조의 경우 순차적으로 입력값을 전달받기 때문에 $t+1$ 시점의 예측을 위해 사용할 수 있는 데이터가 $t$시점까지로 한정된다. **Transformer**는 이러한 구조를 이용하지 않고 전체 입력값을 병렬적으로 전달받기 때문에 과거 시점의 입력값을 예측할 때 미래 시점의 입력값까지 참고할 수가 있다. 이 문제를 방지하기 위한 기법들을 **Look-ahead Mask**라고 하고 이것을 이용하는 Attention을 Masked Attention이라고 한다. 

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F00c04131-3d60-40c0-9032-bdee106d11c9%2FUntitled.png?table=block&id=3633e376-f8d0-4517-8145-8ac7a3e4f361&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

그림과 같이 3개의 입력을 받은 경우를 생각해보자. 
Attention Score 행렬의 (i,j)요소는 i번째 입력값(Query)과 j번째 입력값(Key, Value) 사이의 유사도를 의미한다. 입력값이 순서를 가진 경우 i번째 입력값은 1부터 i까지의 값을 활용할 수 있기 때문에 그림에서 보여지는 행렬의 대각선 윗부분(i<j)는 주어진 입력값이 볼 수 없는 미래시점의 입력값과의 유사도를 의미한다. 이 부분을 가리고 연산을 수행하는것이 Masked Attention인것이다. 구체적으로 보면 Attention Score 행렬의 대각선 윗부분을 -inf로 변경(=Look-ahead Mask)한 후 Softmax를 취해 해당 요소들의 Attention Weight를 0으로 만들어 Attention Value를 계산할 때 미래 시점의 값을 고려하지 않도록 만들어준다.

## (2). Encoder-Decoder Attention

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fbdae7542-2568-45e2-b14a-611db6ea7072%2FUntitled.png?table=block&id=c42fb03d-ad2d-4107-944e-ce91a2267dd2&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

Masked Self-Attention 이후에 등장하는 Attention Layer는 Encoder의 출력값과 Decoder의 입력값을 이용하는 Encoder-Decoder Attention이다. $t$시점의 예측에 도움이 되는 Encoder의 출력값들만 이용하고자 하는 것이 목적으로 Query는 Decoder의 Masked Self-Attention을 통과한 입력값이, Key와 Value는 Encoder의 출력값이 된다.

# Ⅵ. Position-wise FeedForward

Postion-wise feedforward Network는 Self-Attention층을 거친 후 통과하는 네트워크를 말한다. 2개의 선형층을 거치게 되는데, 첫번째 선형층에서 ReLU를 사용한다. 

# Ⅶ. Output Probabilities

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fff9f5f87-152e-440d-b22a-6e35fc607270%2FUntitled.png?table=block&id=50671d01-f087-45f8-877d-cf93b4b127d6&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

이렇게 나온 값들을 Linear Layer와 Softmax Layer를 순차적으로 거치는데
Linear Layer(FC Layer)를 거쳐 Softmax를 태우면 최종적으로 우리가 찾아야하는 단어의 확률값(Softmax Score)를 가지게 되고(vocab_size만큼에 대한 각각의 확률값) 여기서 argmax로 뽑아낸 단어를 최종단어로 선정하게 된다.

![Untitled](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F9adb82e4-dc28-49d8-9465-d75730bb9eba%2FUntitled.png?table=block&id=cfb77963-c704-453a-8879-ebaebb3defc0&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)

# Ⅶ. Transformer : Attention is all you need🔥

![출처 : blossominkyung님 블로그, 트랜스포머 파헤치기 - 1. Positional Encoding](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F1392ec82-a566-4b0e-8dba-ddbcff98f259%2FUntitled.png?table=block&id=6fa9865a-8bdc-48fb-9d43-e0ca649ee560&spaceId=ff68afbb-de24-495e-8075-109156ce3ceb&width=2000&userId=d0eae791-c1ef-435a-9af2-1c48d0457a62&cache=v2)
