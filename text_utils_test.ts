import * as tfc from '@tensorflow/tfjs-core';
import * as text_utils from './text_utils';
import {
  Config, TokenDigitalization, TypeEnumeration, TextTokenization,
  SequenceTransformation, WorkflowIntegration
} from './text_utils';
import { expectTensorsClose } from './test_utils';

describe('text preprocessing utilities', () => {
  let config: Config;
  let corpus: string[];
  beforeEach(() => {
    const text0 = 'word0! \"word1\" (word2)';
    const text1 = 'word3\'s /word4/ `word5 \\word1\\ word6.';
    corpus = [text0, text1];
    config = {
      charactersToFilter: '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
      splittingCharacter: ' ',
      lowerCaseFlag: false,
      maximumNumberOfWords: undefined,
      numberOfTexts: corpus.length,
      characterLevelFlag: false,
      defaultToken: undefined
    };
  });

  describe('TextTokenizer', () => {
    let tokenizer: TextTokenization;
    beforeEach(() => {
      tokenizer = new text_utils.TextTokenization(corpus, config);
    });

    describe('createTextTokenSequence', () => {
      it('should be able to take a text and return an array of tokens',
        () => {
          const textTokenSequence0 =
            tokenizer.createTextTokenSequence(corpus[0], config);
          expect(textTokenSequence0).toEqual(['word0', 'word1', 'word2']);
          const textTokenSequence1 =
            tokenizer.createTextTokenSequence(corpus[1], config);
          expect(textTokenSequence1).toEqual(
            ['word3\'s', 'word4', 'word5', 'word1', 'word6']);
        });
    });

    describe('createTextTokenSequence', () => {
      it('should be able to take a string and' +
        'return an array of characters', () => {
          //const text0 = 'word0 word1 word2'; const text1 = 'word3\'s word4
          //word5, word1 word6,';
          const text0 = 'word0! \"word1\" (word2)';
          const text1 = 'word3\'s /word4/ `word5 \\word1\\ word6.';
          corpus = [text0, text1];
          config = {
            charactersToFilter: '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            splittingCharacter: undefined,
            lowerCaseFlag: false,
            maximumNumberOfWords: undefined,
            numberOfTexts: corpus.length,
            characterLevelFlag: true,
            defaultToken: undefined
          };
          const textTokenSequence0 =
            tokenizer.createTextTokenSequence(corpus[0], config);
          expect(textTokenSequence0).toEqual(['w', 'o', 'r', 'd', '0', '!',
            ' ', '\"', 'w',
            'o', 'r', 'd', '1', '\"', ' ', '(', 'w', 'o', 'r', 'd', '2', ')']);
          const textTokenSequence1 =
            tokenizer.createTextTokenSequence(corpus[1], config);
          expect(textTokenSequence1).toEqual(['w', 'o', 'r', 'd', '3',
            '\'', 's', ' ', '\/', 'w', 'o', 'r', 'd', '4', '\/', ' ', '`',
            'w', 'o', 'r', 'd',
            '5', ' ', '\\', 'w', 'o', 'r', 'd', '1', '\\', ' ', 'w', 'o',
            'r', 'd', '6', '.']);
        });
    });

    describe('collectTokens', () => {
      it('should take an array of strings and return both' +
        'an array of arrays of tokens and an array of tokens in the corpus',
        () => {
          const result = tokenizer.collectTokens(corpus);
          const textTokens = result[0] as string[][];
          const corpusTokens = result[1] as string[];
          expect(textTokens).toEqual([['word0', 'word1', 'word2'],
          ['word3\'s', 'word4', 'word5', 'word1', 'word6']]);
          expect(corpusTokens).toEqual(['word0', 'word1', 'word2',
            'word3\'s', 'word4', 'word5', 'word1', 'word6']);
        });
    });

    describe('collectTypes', () => {
      it('should take an array of array of tokens and return' +
        'an array of arrays of word types, and take an array of' +
        'tokens and returns an array of word types',
        () => {
          const tokens = tokenizer.collectTokens(corpus);
          const textTokens = tokens[0] as string[][];
          const corpusTokens = tokens[1] as string[];
          const result = tokenizer.collectTypes(
            textTokens, corpusTokens);
          const typesByText = result[0] as string[][];
          const corpusTypes = result[1] as string[];
          expect(typesByText).toEqual([['word0', 'word1', 'word2'],
          ['word3\'s', 'word4', 'word5', 'word1', 'word6']]);
          expect(corpusTypes).toEqual(['word0', 'word1', 'word2',
            'word3\'s', 'word4', 'word5', 'word6']);
        });
    });

    describe('createCorpusTokenSequences', () => {
      it('should take an array of strings and' +
        'return an array of arrays of strings',
        () => {
          const corpusTokenSequences =
            tokenizer.createCorpusTokenSequences(corpus, config);
          expect(corpusTokenSequences).toEqual([['word0', 'word1', 'word2'],
          ['word3\'s', 'word4', 'word5', 'word1', 'word6']]);
        });
    });
  });

  describe('TypeEnumeration', () => {
    let enumerator: TypeEnumeration;
    let tokenizer: TextTokenization;
    let textTokens: string[][];
    let corpusTokens: string[];
    let corpusTypes: string[];
    beforeEach(() => {
      enumerator = new TypeEnumeration(corpus, config);
      tokenizer = new text_utils.TextTokenization(corpus, config);
      const result0 = tokenizer.collectTokens(corpus);
      textTokens = result0[0] as string[][];
      corpusTokens = result0[1] as string[];
      const result1 = tokenizer.collectTypes(textTokens, corpusTokens);
      corpusTypes = result1[1] as string[];
    });

    describe('countTypesInText', () => {
    it('should generate a map from each type in the corpus'
      + 'to its absolute frequency in a text', () => {
        const typeCountsInText: Map<string, number> =
          enumerator.countTypesInText(textTokens[0], corpusTypes);
        expect(typeCountsInText.get('word0')).toEqual(1);
        expect(typeCountsInText.get('word1')).toEqual(1);
        expect(typeCountsInText.get('word4')).toEqual(0);
      });
    });

    describe('countTypesByText', () => {
    it('should generate an array of maps, one for each text' +
      'and such that each map is from word types in the text to' +
      'their absolute frequencies in the text',
      () => {
        const typeCountsByText: Array<Map<string, number>> =
          enumerator.countTypesByText(textTokens, corpusTypes);
        expect(typeCountsByText[0].get('word1')).toEqual(1);
        expect(typeCountsByText[1].get('word1')).toEqual(1);
        expect(typeCountsByText[0].get('word0')).toEqual(1);
        expect(typeCountsByText[1].get('word0')).toEqual(0);
      });
    });

      describe('countTypesInCorpus', () => {
        it('should generate a map from a word type to its'
      + 'absolute frequency in the corpus',
      () => {
        const typeCountsInCorpus: Map<string, number>
          = enumerator.countTypesInCorpus(corpusTokens, corpusTypes);
        expect(typeCountsInCorpus.get('word0')).toEqual(1);
        expect(typeCountsInCorpus.get('word1')).toEqual(2);
        expect(typeCountsInCorpus.get('word4')).toEqual(1);
      });
    });

      describe('sortTypesByCount', () => {
        it('takes a mapping from types to counts in the corpus ' +
      'and returns a list of the types sorted ' +
      ' descending by the number of counts', () => {
        const typeCountsInCorpus: Map<string, number>
          = enumerator.countTypesInCorpus(corpusTokens, corpusTypes);
        const typesSortedByCount = enumerator.sortTypesByCount(
          typeCountsInCorpus);
        expect(typesSortedByCount).toEqual(['word1', 'word0', 'word2',
          'word3\'s', 'word4', 'word5', 'word6']);
      });
    });

      describe('createTypeIndex', () => {
        it('returns a mapping from a type to an integer such that ' +
      'the absolute frequency of the type in the corpus determines the '
      + 'integer, words occurring more frequently being assigned'
      + ' smaller integers', () => {
        const typeCountsInCorpus: Map<string, number>
          = enumerator.countTypesInCorpus(corpusTokens, corpusTypes);
        const typesSortedByCount = enumerator.sortTypesByCount(
          typeCountsInCorpus);
        const typeIndex = enumerator.createTypeIndex(
          typeCountsInCorpus, typesSortedByCount, config.defaultToken);
        expect(typeIndex.get('word1')).toEqual(1);
        expect(typeIndex.get('word0')).toEqual(2);
      });
    });

      describe('createTypeIndex with a maximum number of words', () => {
        it('returns a mapping from a type to an integer such that ' +
      'the absolute frequency of the type in the corpus determines the '
      + 'integer, words occurring more frequently being assigned'
      + ' smaller integers, in this configuration with infrequent' +
      'words being replaced by a conventional token', () => {
        config.maximumNumberOfWords=3;
        config.defaultToken='UNK';
        enumerator = new TypeEnumeration(corpus, config);
        tokenizer = new text_utils.TextTokenization(corpus, config);
        const result0 = tokenizer.collectTokens(corpus);
        textTokens = result0[0] as string[][];
        corpusTokens = result0[1] as string[];
        const result1 = tokenizer.collectTypes(textTokens, corpusTokens);
        corpusTypes = result1[1] as string[];
        const typeCountsInCorpus: Map<string, number>
          = enumerator.countTypesInCorpus(corpusTokens, corpusTypes);
        const typesSortedByCount = enumerator.sortTypesByCount(
          typeCountsInCorpus);
        const typeIndex = enumerator.createTypeIndex(
          typeCountsInCorpus, typesSortedByCount, config.defaultToken);
        expect(typeIndex.get('word1')).toEqual(1);
        expect(typeIndex.get('word0')).toEqual(2);
        expect(typeIndex.get('UNK')).toEqual(4);
        
});
});

describe('createReverseTypeIndex', () => {
  it('reverses the injective mapping provided by the type index', () => {
      const typeCountsInCorpus: Map<string, number>
        = enumerator.countTypesInCorpus(corpusTokens, corpusTypes);
      const typesSortedByCount = enumerator.sortTypesByCount(
        typeCountsInCorpus);
      const typeIndex = enumerator.createTypeIndex(
        typeCountsInCorpus, typesSortedByCount, config.defaultToken);
      const reverseTypeIndex = enumerator.createReverseTypeIndex(typeIndex);
      expect(reverseTypeIndex.get(1)).toEqual('word1');
      expect(reverseTypeIndex.get(2)).toEqual('word0');
    });
  });
});

  describe('TokenDigitalization', () => {
    let tokenizer: TextTokenization;
    let enumerator: TypeEnumeration;
    let digitalizer: TokenDigitalization;
    let integerSequences: number[][];
    let typeIndex: Map<string, number>;
    let textTokens: string[][];
    let corpusTokens: string[];
    let corpusTypes: string[];
    let typeCountsInCorpus: Map<string, number>;
    let typesSortedByCount: string[];
    let corpusTokenSequences: string[][];
    let generator: IterableIterator<number[]>;
    beforeEach(() => {
      integerSequences = [];
      tokenizer = new text_utils.TextTokenization(corpus, config);
      enumerator = new TypeEnumeration(corpus, config);
      digitalizer = new text_utils.TokenDigitalization(corpus, config);
      const result0 = tokenizer.collectTokens(corpus);
      textTokens = result0[0] as string[][];
      corpusTokens = result0[1] as string[];
      const result1 = tokenizer.collectTypes(textTokens, corpusTokens);
      corpusTypes = result1[1] as string[];
      typeCountsInCorpus
        = enumerator.countTypesInCorpus(corpusTokens, corpusTypes);
      typesSortedByCount = enumerator.sortTypesByCount(typeCountsInCorpus);
      typeIndex = enumerator.createTypeIndex(
        typeCountsInCorpus, typesSortedByCount, config.defaultToken);
      corpusTokenSequences
        = tokenizer.createCorpusTokenSequences(corpus, config);
      generator =
        digitalizer.generateIntegerSequencesFromCounts(typeIndex, config,
          corpusTokenSequences);
    });



    describe('hashWordsToIntegers', () => {
      const spaceFactor = 20;
      it('should return an array of arrays of integers' +
        'each less than or equal to a specified integer',
        async (done) => {
          integerSequences = await digitalizer.hashWordsToIntegers(
            corpus, spaceFactor, undefined, config);
          for (let i = 0; i < corpus.length; i++) {
          }
          done();          expect(integerSequences.length).toBeGreaterThan(0);
          expect(integerSequences[0][0]).toBeGreaterThanOrEqual(1);
          expect(integerSequences[0][0]).toBeLessThanOrEqual(spaceFactor);
        });
    });

    describe('integerSequenceGenerator', () => {
      it('should define a generator of integer sequences on demand',
        () => {
          const integerSequences: number[][] = [];

          for (const s of generator) {
            integerSequences.push(s);
          }
          expect(integerSequences[0].length).toEqual(3);
          expect(integerSequences[1].length).toEqual(5);
          expect(integerSequences[0][0]).toBeGreaterThanOrEqual(1);
        });

    });


    describe('integerSequenceGenerator', () => {
      it('should define a generator of integer sequences on demand,'+
      'in this case with tokens replaced by a conventional token'+
      'being represented by an integer equal to one plus the maximum'
      +' number of words configuration property',
        () => {
          config.maximumNumberOfWords=3;
          config.defaultToken='UNK';
          tokenizer = new text_utils.TextTokenization(corpus, config);
          enumerator = new TypeEnumeration(corpus, config);
          digitalizer = new text_utils.TokenDigitalization(corpus, config);
          const result0 = tokenizer.collectTokens(corpus);
          textTokens = result0[0] as string[][];
          corpusTokens = result0[1] as string[];
          const result1 = tokenizer.collectTypes(textTokens, corpusTokens);
          corpusTypes = result1[1] as string[];
          typeCountsInCorpus
            = enumerator.countTypesInCorpus(corpusTokens, corpusTypes);
          typesSortedByCount = enumerator.sortTypesByCount(typeCountsInCorpus);
          typeIndex = enumerator.createTypeIndex(
            typeCountsInCorpus, typesSortedByCount, config.defaultToken);
          corpusTokenSequences
            = tokenizer.createCorpusTokenSequences(corpus, config);
          generator =
            digitalizer.generateIntegerSequencesFromCounts(typeIndex, config,
              corpusTokenSequences);    

          const integerSequences: number[][] = [];

          for (const s of generator) {
            integerSequences.push(s);
          }
          expect(integerSequences[0].length).toEqual(3);
          expect(integerSequences[1].length).toEqual(5);
          expect(integerSequences[0][0]).toBeGreaterThanOrEqual(1);
          expect(integerSequences[1][1]).toBeGreaterThanOrEqual(4);
        });

    });
  });

  describe('SequenceTransformation', () => {
    let transformer: SequenceTransformation;
    let tokenizer: TextTokenization;
    let enumerator: TypeEnumeration;
    let digitalizer: TokenDigitalization;
    let textTokens: string[][];
    let corpusTokens: string[];
    let corpusTypes: string[];
    let typeCountsInCorpus: Map<string, number>;
    let typesSortedByCount: string[];
    let typeIndex: Map<string, number>;
    let corpusTokenSequences: string[][];
    let integerSequences: number[][];
    let generator: IterableIterator<number[]>;
    let dataTensor: tfc.Tensor;
    let mode: string;
    let dataMatrix: number[][];
    let outputArrays: boolean;

    beforeEach(() => {
      integerSequences = [];
      tokenizer = new text_utils.TextTokenization(corpus, config);
      enumerator = new TypeEnumeration(corpus, config);
      digitalizer = new text_utils.TokenDigitalization(corpus, config);
      transformer = new SequenceTransformation(config);
      const result0 = tokenizer.collectTokens(corpus);
      textTokens = result0[0] as string[][];
      corpusTokens = result0[1] as string[];
      const result1 = tokenizer.collectTypes(textTokens, corpusTokens);
      corpusTypes = result1[1] as string[];
      typeCountsInCorpus =
        enumerator.countTypesInCorpus(corpusTokens, corpusTypes);
      typesSortedByCount = enumerator.sortTypesByCount(typeCountsInCorpus);
      typeIndex = enumerator.createTypeIndex(
        typeCountsInCorpus, typesSortedByCount, config.defaultToken);
      corpusTokenSequences =
        tokenizer.createCorpusTokenSequences(corpus, config);
      generator =
        digitalizer.generateIntegerSequencesFromCounts(typeIndex, config,
          corpusTokenSequences);

      for (const s of generator) {
        integerSequences.push(s);
      }
    });

    describe('createOccurrenceMatrix', () => {
      it('should construct an occurrence matrix'
        + 'from an array of integer sequences'
        +'using absolute frequencies', () => {
          mode = 'count';
          outputArrays = true;
          const returnValue = transformer.createOccurrenceMatrix(config,
            integerSequences, mode, outputArrays);
          if (returnValue instanceof Array) {
            dataMatrix = returnValue as number[][];
            expect(dataMatrix[0][0]).toEqual(1);
            expect(dataMatrix[1][5]).toEqual(1);
          }
          else {
            dataTensor = returnValue;
            expectTensorsClose(dataTensor.slice([0, 0], [1, 1]),
              tfc.tensor([[1]]));
            expectTensorsClose(dataTensor.slice([1, 5], [1, 1]),
              tfc.tensor([[1]]));
          }
        });
          it('should construct an occurrence matrix from an array of strings'+
          ' using relative frequencies',
            () => {
          mode = 'freq';
          const returnValue = transformer.createOccurrenceMatrix(config,
            integerSequences, mode, outputArrays);
          if (returnValue instanceof Array) {
            dataMatrix = returnValue as number[][];
            expect(dataMatrix[0][0]).toBeCloseTo(0.33);
            expect(dataMatrix[1][5]).toBeCloseTo(0.20);
          }
          else {
            dataTensor = returnValue;
            expectTensorsClose(dataTensor.slice([0, 0], [1, 1]),
              tfc.tensor([[0.33]])
              , 0.01);
            expectTensorsClose(dataTensor.slice([1, 5], [1, 1]),
              tfc.tensor([[0.20]])
              , 0.01);
          }
        });
        it('should construct an occurrence matrix from an array of strings'+
          ' using binary frequencies',
            () => {
          mode = 'binary';
          const returnValue = transformer.createOccurrenceMatrix(config,
            integerSequences, mode, outputArrays);
          if (returnValue instanceof Array) {
            dataMatrix = returnValue as number[][];
            expect(dataMatrix[0][0]).toEqual(1);
            expect(dataMatrix[1][5]).toEqual(1);
          }
          else {
            dataTensor = returnValue;
            expectTensorsClose(dataTensor.slice([0, 0], [1, 1]),
              tfc.tensor([[1]]));
            expectTensorsClose(dataTensor.slice([1, 5], [1, 1]),
              tfc.tensor([[1]]));
          }
        });
        it('should construct an occurrence matrix from an array of strings'+
          ' using tfidf values',
            () => {
              mode = 'tfidf';
          const returnValue = transformer.createOccurrenceMatrix(config,
            integerSequences, mode, outputArrays);
          if (returnValue instanceof Array) {
            dataMatrix = returnValue as number[][];
            expect(dataMatrix[0][0]).toEqual(0);
            expect(dataMatrix[1][5]).toBeCloseTo(0.40);
          }
          else {
            dataTensor = returnValue;
            expectTensorsClose(dataTensor.slice([0, 0], [1, 1]),
              tfc.tensor([[0]])
              , 0.01);
            expectTensorsClose(dataTensor.slice([1, 5], [1, 1]),
              tfc.tensor([[0.40]])
              , 0.01);
          }
        });
      });

    describe('createDataSubsequences', () => {

      it('takes an array of integer arrays and returns '
        + 'an array of arrays of context integers and ' +
        'an array of integers contextualized', () => {
          const windowSize = 3;
          const panePosition = 2;
          const outputArrays = true;
          const result =
            transformer.createSubsequences(integerSequences,
              windowSize,
              panePosition,
              outputArrays
            );
          if (outputArrays) {
            const contextCollector = result[0] as number[][];
            expect(contextCollector[0]).toEqual([2, 3]);
            const objectInContextCollector = result[1] as number[];
            expect(objectInContextCollector[0]).toEqual(1);
          }
          else {
            const contextTensor = result[0] as tfc.Tensor;
            const objectInContextTensor = result[1] as tfc.Tensor;
            expectTensorsClose(contextTensor.slice([0, 0], [1, 2]),
              tfc.tensor([[2, 3]]));
            expectTensorsClose(objectInContextTensor.slice([0], [1]),
              tfc.tensor([1]));
          }
        });
    });

  });
  describe('WorkflowIntegration', () => {
    let tokenizer: TextTokenization;
    let enumerator: TypeEnumeration;
    let digitalizer: TokenDigitalization;
    let integrator: WorkflowIntegration;
    let textTokens: string[][];
    let corpusTokens: string[];
    let corpusTypes: string[];
    let typeCountsInCorpus: Map<string, number>;
    let typesSortedByCount: string[];
    let typeIndex: Map<string, number>;
    let corpusTokenSequences: string[][];
    let integerSequences: number[][];
    let generator: IterableIterator<number[]>;
    let dataTensor: tfc.Tensor;
    let mode: string;
    let dataMatrix: number[][];
    let outputArrays: boolean;

    beforeEach(() => {
      integerSequences = [];
      tokenizer = new text_utils.TextTokenization(corpus, config);
      enumerator = new TypeEnumeration(corpus, config);
      digitalizer = new text_utils.TokenDigitalization(corpus, config);
      integrator = new WorkflowIntegration(corpus, config);
      const result0 = tokenizer.collectTokens(corpus);
      textTokens = result0[0] as string[][];
      corpusTokens = result0[1] as string[];
      const result1 = tokenizer.collectTypes(textTokens, corpusTokens);
      corpusTypes = result1[1] as string[];
      typeCountsInCorpus = enumerator.countTypesInCorpus(
        corpusTokens, corpusTypes);
      typesSortedByCount = enumerator.sortTypesByCount(typeCountsInCorpus);
      typeIndex = enumerator.createTypeIndex(
        typeCountsInCorpus, typesSortedByCount, config.defaultToken);
      corpusTokenSequences = tokenizer.createCorpusTokenSequences(
        corpus, config);
      generator =
        digitalizer.generateIntegerSequencesFromCounts(typeIndex, config,
          corpusTokenSequences);

      for (const s of generator) {
        integerSequences.push(s);
      }
    });

    describe('createOccurrenceMatrixFromStrings with counting', () => {
      const hashingFlag = false;
      //let returnValue;
      it('should construct an occurrence matrix from an array of strings'
      +' using absolute frequencies',
        () => {
          mode = 'count';
          outputArrays = true;
          const returnValue = integrator.createOccurrenceMatrixFromStrings(
            corpus, hashingFlag, mode, outputArrays);
          if (returnValue instanceof Array) {
            dataMatrix = returnValue as number[][];
            expect(dataMatrix[0][0]).toEqual(1);
            expect(dataMatrix[1][5]).toEqual(1);
          }
          else {
            dataTensor = returnValue;
            expectTensorsClose(dataTensor.slice([0, 0], [1, 1]),
              tfc.tensor([[1]]));
            expectTensorsClose(dataTensor.slice([1, 5], [1, 1]),
              tfc.tensor([[1]]));
          }
        });
          it('should construct an occurrence matrix from an array of strings'
          +' using relative frequencies',
            () => {
              mode = 'freq';
          const returnValue = integrator.createOccurrenceMatrixFromStrings(
            corpus, hashingFlag, mode, outputArrays);
          if (returnValue instanceof Array) {
            dataMatrix = returnValue as number[][];
            expect(dataMatrix[0][0]).toBeCloseTo(0.33);
            expect(dataMatrix[1][5]).toBeCloseTo(0.20);
          }
          else {
            dataTensor = returnValue;
            expectTensorsClose(dataTensor.slice([0, 0], [1, 1]),
              tfc.tensor([[0.33]])
              , 0.01);
            expectTensorsClose(dataTensor.slice([1, 5], [1, 1]),
              tfc.tensor([[0.20]])
              , 0.01);
          }
        });
        it('should construct an occurrence matrix from an array of strings'
          +' using binary frequencies',
            () => {
              mode = 'binary';
          const returnValue = integrator.createOccurrenceMatrixFromStrings(
            corpus, hashingFlag, mode, outputArrays);
          if (returnValue instanceof Array) {
            dataMatrix = returnValue as number[][];
            expect(dataMatrix[0][0]).toEqual(1);
            expect(dataMatrix[1][5]).toEqual(1);
          }
          else {
            dataTensor = returnValue;
            expectTensorsClose(dataTensor.slice([0, 0], [1, 1]),
              tfc.tensor([[1]]));
            expectTensorsClose(dataTensor.slice([1, 5], [1, 1]),
              tfc.tensor([[1]]));
          }
        });
        it('should construct an occurrence matrix from an array of strings'
          +' using tfidf values',
            () => {
              mode = 'tfidf';
          const returnValue = integrator.createOccurrenceMatrixFromStrings(
            corpus, hashingFlag, mode, outputArrays);
          if (returnValue instanceof Array) {
            dataMatrix = returnValue as number[][];
            expect(dataMatrix[0][0]).toEqual(0);
            expect(dataMatrix[1][5]).toBeCloseTo(0.40);
          }
          else {
            dataTensor = returnValue;
            expectTensorsClose(dataTensor.slice([0, 0], [1, 1]),
              tfc.tensor([[0]])
              , 0.01);
            expectTensorsClose(dataTensor.slice([1, 5], [1, 1]),
              tfc.tensor([[0.40]])
              , 0.01);
          }
        });
    });

    describe('createOccurrenceMatrixFromStrings with hashing', () => {
      const hashingFlag = true;
      //let returnValue: number[][] | tfc.Tensor;
      it('should construct an occurrence matrix from an array of strings'+
      ' using absolute frequencies',
        async (done) => {
          mode = 'count';
          const returnValue = await
            integrator.createOccurrenceMatrixFromStrings(
              corpus, hashingFlag, mode, outputArrays);
          done();
          if (returnValue instanceof Array) {
            dataMatrix = returnValue as number[][];
            expect(dataMatrix[0][0]).toEqual(1);
            expect(dataMatrix[1][5]).toEqual(1);
          }
          else {
            dataTensor = returnValue;
            expectTensorsClose(dataTensor.slice([0, 0], [1, 1]),
              tfc.tensor([[1]]));
            expectTensorsClose(dataTensor.slice([1, 5], [1, 1]),
              tfc.tensor([[1]]));
          }
        });
      it('should construct an occurrence matrix from an array of strings'+
      ' using relative frequencies',
        async (done) => {
          mode = 'freq';
          const returnValue = await
            integrator.createOccurrenceMatrixFromStrings(
              corpus, hashingFlag, mode, outputArrays);
          done();
          if (returnValue instanceof Array) {
            dataMatrix = returnValue as number[][];
            expect(dataMatrix[0][0]).toBeCloseTo(0.33);
            expect(dataMatrix[1][5]).toBeCloseTo(0.20);
          }
          else {
            dataTensor = returnValue;
            expectTensorsClose(dataTensor.slice([0, 0], [1, 1]),
              tfc.tensor([[0.33]])
              , 0.01);
            expectTensorsClose(dataTensor.slice([1, 5], [1, 1]),
              tfc.tensor([[0.20]])
              , 0.01);
          }
        });
      it('should construct an occurrence matrix from an array of strings'+
      ' using binary frequencies',
        async (done) => {
          mode = 'binary';
          const returnValue = await
            integrator.createOccurrenceMatrixFromStrings(
              corpus, hashingFlag, mode, outputArrays);
          done();
          if (returnValue instanceof Array) {
            dataMatrix = returnValue as number[][];
            expect(dataMatrix[0][0]).toEqual(1);
            expect(dataMatrix[1][5]).toEqual(1);
          } else {
            dataTensor = returnValue;

            expectTensorsClose(dataTensor.slice([0, 0], [1, 1]),
              tfc.tensor([[1]]));
            expectTensorsClose(dataTensor.slice([1, 5], [1, 1]),
              tfc.tensor([[1]]));
          }
        });
      it('should construct an occurrence matrix from an array of strings'+
      ' using tfidf values',
        async (done) => {
          mode = 'tfidf';
          const returnValue = await
            integrator.createOccurrenceMatrixFromStrings(
              corpus, hashingFlag, mode, outputArrays);
          done();
          if (returnValue instanceof Array) {
            dataMatrix = returnValue as number[][];
            expect(dataMatrix[0][0]).toEqual(0);
            expect(dataMatrix[1][5]).toBeCloseTo(0.40);
          } else {
            dataTensor = returnValue;

            expectTensorsClose(dataTensor.slice([0, 0], [1, 1]),
              tfc.tensor([[0]])
              , 0.01);
            expectTensorsClose(dataTensor.slice([1, 5], [1, 1]),
              tfc.tensor([[0.40]])
              , 0.01);
          }
        });
    });

    describe('createDataSubsequences with counting', () => {

      it('takes an array of strings and returns'
        + 'an array of arrays of context integers and'
        + 'an array of integers contextualized', () => {
          const hashingFlag = false;
          const windowSize = 3;
          const panePosition = 2;
          const outputArrays = true;
          const result = integrator.createSubsequencesFromStrings(corpus,
            windowSize,
            panePosition,
            hashingFlag,
            outputArrays
          );
          if (outputArrays) {
            const contextCollector = result[0] as number[][];
            expect(contextCollector[0]).toEqual([2, 3]);
            const objectInContextCollector = result[1] as number[];
            expect(objectInContextCollector[0]).toEqual(1);
          } else {
            const contextTensor = result[0] as tfc.Tensor;
            const objectInContextTensor = result[1] as tfc.Tensor;
            expectTensorsClose(contextTensor.slice([0, 0], [1, 2]),
              tfc.tensor([[2, 3]]));
            expectTensorsClose(objectInContextTensor.slice([0], [1]),
              tfc.tensor([1]));
          }
        });
    });

    describe('createDataSubsequences with hashing', () => {

      it('takes an array of strings and returns' +
        ' an array of arrays of context integers' +
        'and an array of integers contextualized', () => {

          const hashingFlag = true;
          const windowSize = 3;
          const panePosition = 2;
          const outputArrays = true;
          const hashAwaiter = async () => {
            const result =
              await integrator.createSubsequencesFromStrings(corpus,
                windowSize,
                panePosition,
                hashingFlag,
                outputArrays
              );
            if (outputArrays) {
              const contextCollector = result[0] as number[][];
              expect(contextCollector[0]).toEqual([2, 3]);
              const objectInContextCollector = result[1] as number[];
              expect(objectInContextCollector[0]).toEqual(1);
            }
            else {
              const contextTensor = result[0] as tfc.Tensor;
              const objectInContextTensor = result[1] as tfc.Tensor;
              expectTensorsClose(contextTensor.slice([0, 0], [1, 2]),
                tfc.tensor([[2, 3]]));
              expectTensorsClose(objectInContextTensor.slice([0], [1]),
                tfc.tensor([1]));
            }
          };
          hashAwaiter();
        });
    });

  });

});
