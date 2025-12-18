# Simple Named Entity Recognition : Natural Language Script Convert To Formal Order List , McDonald's NER


## Tools :
SpaCy - transition-based sequence learning  
Tok2Vec : Hash embeddings , CNN , contextual token features

## Test Results : 


Text: I'd like 2 Big Mac.  
Entities: [('2', 'QUANTITY'), ('Big Mac', 'PRODUCT')]  
Formatted Order:  
2x Big Mac  
Total: $11.98  

Text: Can I get 1 McChicken and 3 World Famous Fries   Medium?  
Entities: [('1', 'QUANTITY'), ('McChicken', 'PRODUCT'),   ('3', 'QUANTITY'), ('World Famous Fries Medium',   'PRODUCT')]  
Formatted Order:  
1x McChicken  
3x World Famous Fries Medium   
Total: $14.36   


Text: One Big Mac and 2 Soft Drink Large.  
Entities: [('Big Mac', 'PRODUCT'), ('2', 'QUANTITY'),   ('Soft Drink Large', 'PRODUCT')]  
Formatted Order:  
1x Big Mac  
2x Soft Drink Large  
Total: $11.97  
  


## Train Record : 
Epoch 1: Losses: {'ner': np.float32(1089.8408)}  
Epoch 2: Losses: {'ner': np.float32(591.7071)}   
Epoch 3: Losses: {'ner': np.float32(187.53241)}   
Epoch 4: Losses: {'ner': np.float32(48.66957)}   
Epoch 5: Losses: {'ner': np.float32(8.190075)}   
Epoch 6: Losses: {'ner': np.float32(7.4448876)}  
Epoch 7: Losses: {'ner': np.float32(5.0094986)}  
Epoch 8: Losses: {'ner': np.float32(0.34239775)}  
Epoch 9: Losses: {'ner': np.float32(2.9335117)}  
Epoch 10: Losses: {'ner': np.float32(1.6297954)}  
Epoch 11: Losses: {'ner': np.float32(0.35869145)}  
Epoch 12: Losses: {'ner': np.float32(1.7466378)}  
Epoch 13: Losses: {'ner': np.float32(0.0072427616)}  
Epoch 14: Losses: {'ner': np.float32(3.585416)}  
Epoch 15: Losses: {'ner': np.float32(0.0079072695)}  
Epoch 16: Losses: {'ner': np.float32(0.00031384928)}  
Epoch 17: Losses: {'ner': np.float32(0.016193599)}  
Epoch 18: Losses: {'ner': np.float32(0.054577082)}  
Epoch 19: Losses: {'ner': np.float32(0.039379604)}  
Epoch 20: Losses: {'ner': np.float32(0.017192394)}  


