## 특허 기술 분류
효율적인 문서 분류 자동화를 위해 많은 연구들이 이루어지고 있다. 이 중 특허문서
분류는 서지 정보와 기술 내용이 공존한다는 점에서 일반 문서의 분류보다 상대적으로
복잡할 뿐만 아니라 전문적인 지식과 많은 시간이 요구된다. 본 연구에서는 특허문서에
포함된 각 서지 정보의 특성과 텍스트형 정보의 특성에 적합한 전처리 과정을 통해
특허 정보를 벡터화하고 이를 딥러닝 기반의 모델에 적용시켜 특허문서 기술분류의
정확도를 보다 높일 수 있는 방법론을 제시한다

### Phase1

#### Dataset
- 검색식   *검색 결과 2137개*
<br>(방독면 정화통 여과통 ((가스* 방독* gas*) near (마스크 mask)) respirator 캐니스터
  카니스터 canister) and (마이크 마이크로폰 mike microphone 안경 glass* 렌즈 lens
필터 filter 카트리지 cartridge ((흡기* intak* inhal* 배기* 호기* exhalat* exhaust*) near
(유닛 유니트 unit)) (active* near (carbon charcoal)) 활성탄)).ti,ab,cla. And (A62B*
B01D*).

- 필터링 기준  *필터링 후  62개* <br>

| 등록번호  | 유효여부  | 국가코드  |
  |--- |---|---|
  | 유  | 유효  | US  |

- Feature 선택
청구항 수 , 출원인 대표명화 명칭(영문) , 발명자/고안자 , 우선권 번호 ,Original IPC All

##### 추가예정
---------------------------------



### Phase 2

#### Dataset
- 데이터셋 추가 방안
 - 특허나라 특허 동향 보고서의 검색식을 기반으로 WIPS 데이터베이스에서 추출(기술분류 완료됨)
 - 특허나라 유효특허 목록과 대조하여 유효한 특허만 추출

- 추가된 데이터셋
 - 방사성 동위원소 생산 (이하 radio) - 104개
 - 선박 (이하 ship) - 88개
 - 방독면 (이하 mask) - 62개

- Feature 선택
 abstract, claim, title, Original IPC All

#### Data Preprocessing
 - code 참고

#### Method
 1. text embedding(Doc2vec, Word2vec, GloVe)
 2. SMOTE (데이터셋 특정 라벨의 개수가 적어 overfitting 가능성 큼, ppt 첨부)
 3. Inception model 적용



 <br>
 ##### 추가예정
------------------------------------------------
