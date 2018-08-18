/**
 * @license
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import * as tfc from '@tensorflow/tfjs-core';

/**
 * Collection of classes with methods largely modelled after the Keras utilities
 * for text input preprocessing.
 */

/**
 *  Defines the parameters which are passed to the classes. 
 */
export interface Config {
  /**
   * @property String of characters each of which should be deleted from the
   * texts before tokenization.
   */
  charactersToFilter: string;
  /**
   * @property The character at which texts are divided into tokens if
   * characterLevelFlag should not be true.
   */
  splittingCharacter: string;
  /**
   * @property Flag for whether case should be ignored.
   */
  lowerCaseFlag: boolean;
  /**
   * @property The maximum number of words for which characteristics should be
   * collected.
   */
  maximumNumberOfWords: number;
  /**
   * @property How many documents are in the corpus.
   */
  numberOfTexts: number;
  /**
   * @property Flag for whether each letter is a token or whether tokens are
   * separated by spitting symbols. 
   */
  characterLevelFlag: boolean;
  /**
   * @property Conventional token replacing each word which the number-of-words
   * limit forces exclusion of.
   */
  defaultToken: string;
}

/**
 * Collection of methods for splitting the texts of a corpus of texts into
 * arrays of strings.
 */
export class TextTokenization implements Config {

  config: Config;
  corpus: string[];
  charactersToFilter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n';
  splittingCharacter = ' ';
  lowerCaseFlag = true;
  maximumNumberOfWords: number;
  numberOfTexts: number;
  characterLevelFlag = false;
  defaultToken: string;

  /**
   * @param corpus Array of texts as strings.
   * @param config Class configuration.
   */
  constructor(corpus: string[], config: Config) {
    this.config = config;
    this.corpus = corpus;
    this.charactersToFilter = config.charactersToFilter;
    this.splittingCharacter = config.splittingCharacter;
    this.lowerCaseFlag = config.lowerCaseFlag;
    this.maximumNumberOfWords = config.maximumNumberOfWords;
    this.numberOfTexts = config.numberOfTexts;
    this.characterLevelFlag = config.characterLevelFlag;
    this.defaultToken = config.defaultToken;
  }

  /**
   * @param text Text given as a string. 
   * @param config Class configuration.
   * @returns Array of strings which are the text's tokens.
   */
  createTextTokenSequence(text: string, config: Config): string[] {

    let tokens: string[];

    /* use empty string as splitting character to effect processing of single
    characters as tokens */
    if (config.characterLevelFlag) {
      config.splittingCharacter = '';
    }

    if (config.lowerCaseFlag) {
      text = text.toLowerCase();
    }

    if (config.charactersToFilter) {
      let charactersToFilterArray;
      /*
       *  rewrite charactersToFilter as array if given as a string
       */
      if (typeof (config.charactersToFilter) === 'string') {
        charactersToFilterArray = config.charactersToFilter.split('');
      }
      else {
        charactersToFilterArray = config.charactersToFilter;
      }
      /*
       * build a regular expression enabling both the split character and any
       * filter character to split the text
       */
      /* a holder for the regex pattern to be constructed */
      let charactersToFilterRegExpPrep = '';
      /* append a bracket to the empty holder*/
      charactersToFilterRegExpPrep = charactersToFilterRegExpPrep + '[';
      /*
       * escape any characters to be filtered which are functional in regular
       * expression syntax
       */
      charactersToFilterArray.forEach((f) => {
        if ('[]|.\\+*?{},$^=!\"\`'.includes(f)) {
          charactersToFilterRegExpPrep =
            charactersToFilterRegExpPrep + '\\' + f;
        }
        else {
          charactersToFilterRegExpPrep =
            charactersToFilterRegExpPrep + f;
        }
      });
      /* complete the pattern with a bracket */
      charactersToFilterRegExpPrep =
        charactersToFilterRegExpPrep + config.splittingCharacter + ']';

      const charactersToFilterRegExp = RegExp(charactersToFilterRegExpPrep);
      if (config.splittingCharacter !== '') {
        /* if splitting should not be at character level, then split by any
        character in the regular expression */
        tokens = text.split(charactersToFilterRegExp);
      }
      else {
        /* otherwise split at the empty string */
        tokens = text.split('');
      }
      /*
       * eliminate empty strings possibly induced by splitting
       */
      tokens = tokens.filter((f) => f !== '');
    }
    return tokens;
  }

  /**
   * @param corpus Array of strings representing texts.
   * @param config Class configuration.
   * @returns Array of arrays of strings which are the tokens in those texts.
   */
  createCorpusTokenSequences(corpus: string[], config: Config): string[][] {
    const corpusTokenSequences: string[][] = [];
    for (let text = 0; text < corpus.length; text++) {
      corpusTokenSequences.push(
        this.createTextTokenSequence(corpus[text], config));
    }
    return corpusTokenSequences;
  }

  /**
   * @param corpus Array of texts as strings.
   * @returns Array of arrays of tokens in each text and an array of tokens in
   * the corpus.
   */
  collectTokens(corpus: string[]): [string[][], string[]] {
    const textTokens: string[][] = [];
    let corpusTokens: string[] = [];
    for (let text = 0; text < corpus.length; text++) {
      /*
       * produce an array of tokens per text and append it to the array for
       * texts
       */
      textTokens.push(this.createTextTokenSequence(corpus[text], this.config));
      /*
       * append each array to the corpus array
       */
      corpusTokens =
        corpusTokens.concat(textTokens[text]);
    }
    return [textTokens, corpusTokens];
  }

  /**
   * @param textTokens Array of arrays of tokens, one for each text in the
   * corpus.
   * @param corpusTokens Array of tokens in the corpus.
   * @returns Array of array of word types in each text and an array of word
   * types in the corpus.
   */
  collectTypes(textTokens: string[][], corpusTokens: string[]):
    [string[][], string[]] {
    const textTypes: string[][] = [];
    let corpusTypes: string[] = [];
    for (let c = 0; c < textTokens.length; c++) {
      /*
       * take the arrays of tokens and create arrays without duplicates
       */
      textTypes.push(Array.from(new Set(textTokens[c])));
      /*
       * append contents of each to an array for the entire corpus while
       * eliminating possible duplications across texts
       */
      corpusTypes = Array.from(new Set(corpusTypes.concat(textTypes[c])));
    }
    /*
     * if a maximum number of words has not been defined let the number of types
     * in the corpus be that number
     */
    if (this.config.maximumNumberOfWords === undefined) {
      this.config.maximumNumberOfWords = corpusTypes.length;
    }
    return [textTypes, corpusTypes];
  }

}

/**
 * Collection of methods for determining absolute frequencies of word types.
 */
export class TypeEnumeration implements Config {

  config: Config;
  corpus: string[];

  charactersToFilter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n';
  splittingCharacter = ' ';
  lowerCaseFlag = true;
  maximumNumberOfWords = 0;
  numberOfTexts = 0;
  characterLevelFlag = false;
  defaultToken = 'UNK';

  /**
   * @param corpus Array of texts as strings.
   * @param config Configuration of class.
   */
  constructor(corpus: string[], config: Config) {
    this.config = config;
    this.corpus = corpus;
    this.charactersToFilter = config.charactersToFilter;
    this.splittingCharacter = config.splittingCharacter;
    this.lowerCaseFlag = config.lowerCaseFlag;
    this.maximumNumberOfWords = config.maximumNumberOfWords;
    this.numberOfTexts = config.numberOfTexts;
    this.characterLevelFlag = config.characterLevelFlag;
    this.defaultToken = config.defaultToken;
  }

  /** 
   * @param tokensInText Array of tokens in a text. 
   * @param corpusTypes Array of types in the corpus of texts which includes
   * that text.
   * @returns Map from a type to the number of times it occurs in the text.
   */
  countTypesInText(tokensInText: string[], corpusTypes: string[]):
    Map<string, number> {
    const typeCountsInText = new Map<string, number>();

    /*
     * for each type in the corpus create an array of its occurrences in a text
     * and determine the length of this array
     */
    for (let i = 0; i < corpusTypes.length; ++i) {
      const reducedList = tokensInText.filter(w => w === corpusTypes[i]);
      typeCountsInText.set(corpusTypes[i], reducedList.length);
    }
    return typeCountsInText;
  }

  /**
   * @param tokensByText Array of arrays of tokens in each text of a corpus.
   * @param corpusTypes Array of word types in the corpus.
   * @returns Array of maps, one for each text, from the word type of each token
   * in that text to the number of times that type occurs as a token in the
   * respective text.
   */
  countTypesByText(tokensByText: string[][], corpusTypes: string[]):
    Array<Map<string, number>> {
    const typeCountsByText: Array<Map<string, number>> = [];
    for (let i = 0; i < tokensByText.length; i++) {
      typeCountsByText.push(this.countTypesInText(tokensByText[i],
        corpusTypes));
    }
    return typeCountsByText;
  }

  /**
   * @param tokensinCorpus Array of the tokens in a corpus.
   * @param corpusTypes Array of the word types in the corpus.
   * @returns Map from the type of each token to the number of times the type
   * occurs as a token in the corpus.
   */
  countTypesInCorpus(tokensInCorpus: string[], corpusTypes: string[]):
    Map<string, number> {
    const typeCountsInCorpus: Map<string, number>
      = this.countTypesInText(tokensInCorpus, corpusTypes);
    return typeCountsInCorpus;
  }

  /**
   * @param typeCountsInCorpus Map from types to counts in a corpus
   * @returns Array of the types sorted in descending order according to the
   * magnitude of the counts.
   */
  sortTypesByCount(typeCountsInCorpus: Map<string, number>): string[] {
    const typeCountsInCorpusArray = Array.from(typeCountsInCorpus.entries());
    /*
    * taking counts from two elements of the array determine their difference
    * and sort by its sign, creating a descending order
    */
    typeCountsInCorpusArray.sort((onePair: [string, number],
      anotherPair: [string, number]) =>
      anotherPair[1] - onePair[1]);
    /*
  * create a corresponding sorted array of the words which have been counted
  * */
    const typesSortedByCount
      = Array.from(typeCountsInCorpusArray.map((p) => p[0]));
    return typesSortedByCount;
  }

  /** 
   * @param typeCountsInCorpus Map from word types to their counts in the
   * corpus. 
   * @param typesSortedByCount Types in descending order by frequency.
   * @param defaultToken Default token for the corpus.
   * @returns Map from a type to an integer, such that the absolute frequency of
   *  the word in the corpus determines the integer, words occurring more
   *  frequently being assigned smaller integers.
   */
  createTypeIndex(typeCountsInCorpus:
    Map<string, number>, typesSortedByCount: string[], defaultToken: string):
    Map<string, number> {

    const typeIndex = new Map<string, number>();
    /*
    *  traverse the list of sorted types up to a maximum and associate with each
    *  type an integer beginning with 1
    */
    for (let i = 0; i < this.config.maximumNumberOfWords; i++) {
      typeIndex.set(typesSortedByCount[i], i + 1);
    }
    if (defaultToken !== undefined) {
      /* if a default token has been defined set the maximum number of words 
      * plus one as an integer corresponding to the default token in the mapping
      * from type to integer
      */
      typeIndex.set(defaultToken, this.config.maximumNumberOfWords + 1);
      /*
      * set the count for this ficticous type as the count for the default token
      * in the mapping from types to counts
      */
    }
    return typeIndex;
  }

  /**
   * @param typeIndex Map from type to integer
   * @returns Map from integer to type
   */
  createReverseTypeIndex(typeIndex: Map<string, number>): Map<number, string> {
    const types = typeIndex.keys();
    const indices = typeIndex.values();
    const reverseTypeIndex = new Map<number, string>();
    let index: IteratorResult<number>;
    let type: IteratorResult<string>;
    /*
    *  take a list of keys and a list of values from the first mapping, set each
    *  value as key and key as value in the second mapping
    */
    do {
      index = indices.next();
      type = types.next();
      /* set key and value if the end of the list has not been reached
      */
      if (index.done !== true) {
        reverseTypeIndex.set(index.value, type.value);
      }
    } while (index.done !== true);

    return reverseTypeIndex;
  }
}

/**
 * Class of two asynchronous methods which when given texts, types and tokens
 * creates sequences of integers in place of tokens.
 */
export class TokenDigitalization implements Config {

  config: Config;
  corpus: string[];

  charactersToFilter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n';
  splittingCharacter = ' ';
  lowerCaseFlag = true;
  maximumNumberOfWords = 0;
  numberOfTexts = 0;
  characterLevelFlag = false;
  defaultToken = 'UNK';

  /**
   * @param corpus Array of texts as strings.
   * @param config Class configuration.
   */
  constructor(corpus: string[], config: Config) {
    this.config = config;
    this.corpus = corpus;
    this.charactersToFilter = config.charactersToFilter;
    this.splittingCharacter = config.splittingCharacter;
    this.lowerCaseFlag = config.lowerCaseFlag;
    this.maximumNumberOfWords = config.maximumNumberOfWords;
    this.numberOfTexts = config.numberOfTexts;
    this.characterLevelFlag = config.characterLevelFlag;
    this.defaultToken = config.defaultToken;
  }

  /**
   * @param corpus Array of texts as strings.
   * @param spaceFactor Maximum limit of hashing region.
   * @param config Class configuration.
   * @returns Array of arrays of integers less than or equal to a magnitude
   * given as a parameter and such that each integer corresponds to a word type
   * in a text.
   */
  hashWordsToIntegers = async (corpus: string[], spaceFactor: number,
     config: Config): Promise<number[][]> => {

    const tokenizer = new TextTokenization(corpus, config);

    /* this will be an array with all tokens in the corpus*/
    let totalSequence: string[] = [];
    /*
    *  the beginning and end of each text will be recorded so that the
    *  corresponding integer sequences can be deduced from the total sequence
    *  for the corpus; a zero is preappended as a starting boundary
    */
    const textBoundaries: number[] = [0];
    /*
    * * create a sequence of tokens for each text in the corpus and concatenate
    *   it to the corpus sequence of tokens
    */
    for (let i = 0; i < corpus.length; i++) {
      const partialSequence =
        tokenizer.createTextTokenSequence(corpus[i], config);
      /*
      * add the length of the sequence to the previous boundary to obtain the
      * boundary for this text
      */
      textBoundaries.push(textBoundaries[i] + partialSequence.length);
      totalSequence = totalSequence.concat(partialSequence);
    }

    // function from polyfill at
    // https://developer.mozilla.org/en-US/docs/Web/API/TextEncoder
    function encode(str: string) {
      const lLen = str.length;
      let resPos = -1;
      let resArr = new Uint8Array(lLen * 3);
      for (let point = 0, nextcode = 0, i = 0; i !== lLen;) {
        point = str.charCodeAt(i), i += 1;
        if (point >= 0xD800 && point <= 0xDBFF) {
          if (i === lLen) {
            resArr[resPos += 1] = 0xef/*0b11101111*/;
            resArr[resPos += 1] = 0xbf/*0b10111111*/;
            resArr[resPos += 1] = 0xbd/*0b10111101*/;
            break;
          }
          // https://mathiasbynens.be/notes/
          // javascript-encoding#surrogate-formulae
          nextcode = str.charCodeAt(i);
          if (nextcode >= 0xDC00 && nextcode <= 0xDFFF) {
            point = (point - 0xD800) * 0x400 + nextcode - 0xDC00 + 0x10000;
            i += 1;
            if (point > 0xffff) {
              const encoder = new TextEncoder();
              encoder.encode();

              resArr[resPos += 1] = (0x1e/*0b11110*/ << 3) |
                (point >>> 18);
              resArr[resPos += 1] = (0x2/*0b10*/ << 6) |
                ((point >>> 12) & 0x3f/*0b00111111*/);
              resArr[resPos += 1] = (0x2/*0b10*/ << 6) |
                ((point >>> 6) & 0x3f/*0b00111111*/);
              resArr[resPos += 1] = (0x2/*0b10*/ << 6) |
                (point & 0x3f/*0b00111111*/);
              continue;
            }
          } else {
            resArr[resPos += 1] = 0xef/*0b11101111*/;
            resArr[resPos += 1] = 0xbf/*0b10111111*/;
            resArr[resPos += 1] = 0xbd/*0b10111101*/;
            continue;
          }
        }
        if (point <= 0x007f) {
          resArr[resPos += 1] = (0x0/*0b0*/ << 7) |
            point;
        } else if (point <= 0x07ff) {
          resArr[resPos += 1] = (0x6/*0b110*/ << 5) |
            (point >>> 6);
          resArr[resPos += 1] = (0x2/*0b10*/ << 6) |
            (point & 0x3f/*0b00111111*/);
        } else {
          resArr[resPos += 1] = (0xe/*0b1110*/ << 4) |
            (point >>> 12);
          resArr[resPos += 1] = (0x2/*0b10*/ << 6) |
            ((point >>> 6) & 0x3f/*0b00111111*/);
          resArr[resPos += 1] = (0x2/*0b10*/ << 6) |
            (point & 0x3f/*0b00111111*/);
        }
      }
      resArr = new Uint8Array(resArr.buffer.slice(0, resPos + 1));
      return resArr;
    }

    // from example at
    //https://developer.mozilla.org/en-US/docs/Web/API/SubtleCrypto/digest
    async function sha(message: string) {

      // encode as UTF-8
      ///const msgBuffer = new TextEncoder().encode(message, 'utf-8');
      const msgBuffer = encode(message);

      // hash the message
      const hashBuffer = await crypto.subtle.digest('SHA-1', msgBuffer);

      //convert ArrayBuffer to Array
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      const hashHex = hashArray.map(b =>
        ('00' + b.toString(16)).slice(-2)).join('');
      return hashHex;
    }

    const integerSequence: number[] = [];
    /* apply the hash function to each word in the corpus */
    for (let m = 0; m < totalSequence.length; m++) {
      const value =
        parseInt(await sha(
          totalSequence[m]), 16) % (spaceFactor - 1) + 1;
      integerSequence.push(value);
    }

    /*
     * an array of arrays of integers for the sequences in the texts of the
     * corpus
    */
    const integerSequences: number[][] = [];
    /*
     * use the text boundaries recorded to slice the total integer sequences
     * into sequences for each text
    */
    for (let i = 0; i < textBoundaries.length - 1; i++) {
      integerSequences.push(
        integerSequence.slice(textBoundaries[i], textBoundaries[i + 1]));
    }
    return integerSequences;
  };

  /**
   * @param typeIndex Map from word types to integers.
   * @param config Class configuration.
   * @param corpusTokenSequences Array of arrays of token as strings.
   * @returns Iterator of arrays of integers such that each integer corresponds
   * to the word type of the token at a point in a text 
   */
  generateIntegerSequencesFromCounts = function* (
    typeIndex: Map<string, number>, config: Config,
    corpusTokenSequences: string[][]): IterableIterator<number[]> {
    /* attempt to retrieve the integer associated with each token */
    for (let t = 0; t < corpusTokenSequences.length; t++) {
      const tokenSequence = corpusTokenSequences[t];
      const integerSequence = [];
      for (let w = 0; w < tokenSequence.length; w++) {
        let i = typeIndex.get(tokenSequence[w]);
        if (i !== undefined) {
          /* if it is defined then if maximumNumberOfWords has been defined and
          * if the integer is larger than maximumNumberOfWords ignore the
          * integer
          */
          if (config.maximumNumberOfWords && i > config.maximumNumberOfWords) {
            continue;
          }
          /* otherwise insert the integer into the integer sequence */
          else {
            integerSequence.push(i);
          }
        }
        else {
          /*
          * otherwise if the token is not defined in the word index then if an
          * out of vocabulary token has been defined
            */
          /*
          * retrieve the default token and insert that token's integer into
          * integerArray
          */
          i = typeIndex.get(config.defaultToken);
          if (i != null) {
            integerSequence.push(i);
          }
        }
      }
      yield integerSequence;
    }
  };

}

/**
 * Class of methods for creating an occurrence
 * matrix or context subsequences from arrays of integers.
 */
export class SequenceTransformation implements Config {

  config: Config;

  charactersToFilter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n';
  splittingCharacter = ' ';
  lowerCaseFlag = true;
  maximumNumberOfWords = 0;
  numberOfTexts = 0;
  characterLevelFlag = false;
  defaultToken = 'UNK';

  /**
   * @param config Class configuration.
   */
  constructor(config: Config) {
    this.config = config;
    this.charactersToFilter = config.charactersToFilter;
    this.splittingCharacter = config.splittingCharacter;
    this.lowerCaseFlag = config.lowerCaseFlag;
    this.maximumNumberOfWords = config.maximumNumberOfWords;
    this.numberOfTexts = config.numberOfTexts;
    this.characterLevelFlag = config.characterLevelFlag;
    this.defaultToken = config.defaultToken;
  }


  /** 
   * @param config Class configuration.
   * @param integerSequences Array of integer sequences each representing a
   * text.
   * @param mode Statistic to be inserted in matrix cells.
   * @param outputArrays Flag for whether an array of arrays should be returned
   * instead of a two-dimensional tensor.
   * @returns Array of arrays of integers interpretable as an occurence matrix,
   * with rows corresponding to texts and columns corresponding to features.
   */
  createOccurrenceMatrix(
    config: Config,
    integerSequences: number[][],
    mode: string,
    outputArrays = false
  ): number[][] | tfc.Tensor<tfc.Rank.R2> {
    const dataMatrix: number[][] = [];
    let dataBuffer: tfc.TensorBuffer<tfc.Rank.R2>;

    /*
    * a loop which takes an array of integer sequences, one for each text, and
    *  creates a mapping from an integer representing a word type to the number
    * of texts in which the type occurs
    */
    const textCountPerWordType = new Map<number, number>();
    for (let i = 0; i < integerSequences.length; i++) {
      /*
      * let wordTypeIntegers be the set derived from a list of integer
      * representations of the tokens in a text
      */
      const wordTypeIntegers: number[] =
        Array.from(new Set(integerSequences[i]));

      wordTypeIntegers.forEach(testElement => {
        /*
        * for each element in that set
        * if the word type has not been registered yet
        * set the count to 1
        * otherwise augment its count by 1
        */
        if (textCountPerWordType.get(testElement) === undefined) {
          textCountPerWordType.set(testElement, 1);
        }
        else {
          textCountPerWordType.set(testElement,
            textCountPerWordType.get(testElement) + 1);
        }
      });
    }

    if (outputArrays) {
      /*
      * if an array of arrays should be returned, an array of number arrays is
      * initialized into which values will be inserted at relevant positions
      */
      for (let i = 0; i < integerSequences.length; i++) {
        dataMatrix.push(new Array<number>(config.maximumNumberOfWords).fill(0));
      }
    }
    else {
      /* otherwise a tensor data buffer is provided in which entries will be
      * made
      */
      dataBuffer = tfc.buffer(
        [integerSequences.length, config.maximumNumberOfWords]);
    }
    /* an enumerator which enumerators the sequences */
    const sequenceEnumerator = integerSequences.entries();
    /* a flag indicating whether the enumeration has been completed */
    let quit = false;
    let test = sequenceEnumerator.next();
    /* retrieve the first sequence and repeat the following until the quit flag
    has turned true */
    do {
      /*
      * the first part of the enumerated value is an index representing the text
      * being processed 
      */
      const textInteger = test.value[0];
      /* the second part is the sequence of integers representing word types */
      const integerSequence = test.value[1];
      /*
      * create a mapping from the integers representing word types and counts of
      * their occurrences in the integer sequences
      */
      const integerCounts = new Map<number, number>();
      /*
      * since an integer has been assigned consecutively to each word and since
      * the integer reflects its frequency in the corpus, with the most frequent
      * words being assigned the lowest integers, those words with an integer
      * exceeding maximumNumberOfWords can be passed over by exiting the loop
      */
      for (let i = 0; i < integerSequence.length; i++) {
        const j = integerSequence[i];
        /* exit if the word limit has been exceeded */
        if (j > config.maximumNumberOfWords) { continue; }
        /*
        * if it is not already in the domain of the mapping set its count to 1
        */
        if (integerCounts.get(j) === undefined) { integerCounts.set(j, 1); }
        /* 
        * if it is already in the domain increase its count by 1
        */
        else { integerCounts.set(j, integerCounts.get(j) + 1); }
      }

      /* for each pair of a type count and the type's representative integer */
      integerCounts.forEach((count, typeInteger) => {
        if (mode === 'count') {
          /*
          * if the matrix should consist of absolute frequencies make the type
          * count the matrix content at row textInteger and column
          * featureInteger less one
          */
          if (outputArrays) {
            dataMatrix[textInteger][typeInteger - 1] = count;
          }
          else {
            dataBuffer.set(count, textInteger, typeInteger - 1);
          }
        }
        if (mode === 'freq') {
          /*
          * if the matrix should consist of relative frequencies make the ratio
          * of count and text length the matrix content at row textIndex and
          * column typeInteger less one
          */
          if (outputArrays) {
            dataMatrix[textInteger][typeInteger - 1] =
              count / integerSequence.length;
          }
          else {
            dataBuffer.set(count / integerSequence.length,
              textInteger, typeInteger - 1);
          }
        }
        if (mode === 'binary') {
          /*
          * if the matrix should consist of binary frequencies representing
          * presence or absence of a word in a text set a one as the matrix
          * content at row textInteger and column typeInteger less one
          */
          if (outputArrays) {
            dataMatrix[textInteger][typeInteger - 1] = 1;
          }
          else {
            dataBuffer.set(1,
              textInteger, typeInteger - 1);
          }
        }
        if (mode === 'tfidf') {
          /*
          * if the matrix should consist of tfidf values calculate a document
          * term weight, a logarithm of the absolute frequency of the word in
          * the text augmented by one
          */
          const tf = 1 + Math.log(count);
          /*
          * and calculate a query term weight, the logarithm of the quotient of
          * one plus the number of texts and the number of texts containing the
          * word type plus one
          */
          const idf = Math.log((1 + config.numberOfTexts) /
            (1 + textCountPerWordType.get(typeInteger)));
          /*
        * then set the product of the document term weight and the query term
        * weight as the content of the matrix at row textinteger and column
        * typeInteger less one
        */
          if (outputArrays) {
            dataMatrix[textInteger][typeInteger - 1] = tf * idf;
          } else {
            dataBuffer.set(tf * idf,
              textInteger, typeInteger - 1);
          }
        }
      });
      test = sequenceEnumerator.next();
      /*
      retrieve data for the next possible sequence and
      * set the quit flag according to the enumerator's finished flag
      */
      quit = test.done;
    } while (quit === false);

    if (outputArrays) {
      return dataMatrix;
    } else {
      return dataBuffer.toTensor();
    }
  }

  /**
   * @param integerSequences Arrays of integers representing texts.
   * @param windowSize Length of the subsequences to be processed.
   * @param panePosition Position in the window of the token the context of
   * which is to be extracted.
   * @returns Array of integers which are the contexts of integers in a seqences
   * and an array of those corresponding integers.
   */
  createSubsequences(integerSequences:
    number[][],
    windowSize: number,
    panePosition: number,
    outputArrays = false
  ): [number[][], number[]] | [tfc.Tensor<tfc.Rank.R2>,
    tfc.Tensor<tfc.Rank.R1>] {

    /* the breadth of the window frame to the left of the token */
    const leftSideBreadth = panePosition - 1;
    /* the breadth of the window frame to the right of the token */
    const rightSideBreadth = windowSize - panePosition;
    /*
    * set a pointer to the index of the token's point in an integer sequence,
    * which is initially the length to the sequence to the left
    */
    let pointer = leftSideBreadth;

    const contextCollector: number[][] = [];
    const objectInContextCollector: number[] = [];
    let contextTensor: tfc.Tensor<tfc.Rank.R2>;
    let objectInContextTensor: tfc.Tensor<tfc.Rank.R1>;

    let textCounter = 0;
    while (textCounter < this.config.numberOfTexts) {
      /*
      * until just before the pointer plus the right side of the window would
      * place the window beyond the integer sequence
      */
      while (pointer + rightSideBreadth <
        integerSequences[textCounter].length
      ) {

        /*
        * for each point in the window retrieve the appropriate integer from the
        * integer sequence
        */
        const subsequenceWindow = integerSequences[textCounter]
          .slice(pointer - leftSideBreadth, pointer + 1 + rightSideBreadth);
        /* retrieve the integer of the token to be observed */
        const contextualizedObject = integerSequences[textCounter][pointer];
        /* build an array like subsequenceWindow but with the object's integer 
        * omitted
        */
        const context = subsequenceWindow.slice(0, leftSideBreadth)
          .concat(subsequenceWindow.slice(leftSideBreadth + 1, windowSize));
        /* add this context to the collection of contexts */
        contextCollector.push(context);
        /* add this object to the collection of contextualized objects */
        objectInContextCollector.push(contextualizedObject);
        /* move window forward */
        pointer = pointer + 1;
      }
      /* proceed to next text in the corpus */
      textCounter = textCounter + 1;
      /* resetting the pointer */
      pointer = leftSideBreadth;
    }


    if (outputArrays) {
      return [contextCollector, objectInContextCollector];
    }
    else {
      contextTensor = tfc.tensor(contextCollector);
      objectInContextTensor = tfc.tensor(objectInContextCollector);
      return [contextTensor, objectInContextTensor];
    }
  }
}

/**
 * Methods which chain the methods provided by the classes TextTokenization,
 * TypeEnumeration, TokenDigitalization, and SequenceTransformation into
 * workflows from a corpus of texts as strings to tensors.
 */
export class WorkflowIntegration implements Config {

  config: Config;
  corpus: string[];
  charactersToFilter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n';
  splittingCharacter = ' ';
  lowerCaseFlag = true;
  maximumNumberOfWords = 0;
  numberOfTexts = 0;
  characterLevelFlag = false;
  defaultToken = 'UNK';

  /**
   * @param corpus Array of texts as strings.
   * @param config Class configuration.
   */
  constructor(corpus: string[], config: Config) {
    this.config = config;
    this.corpus = corpus;
    this.charactersToFilter = config.charactersToFilter;
    this.splittingCharacter = config.splittingCharacter;
    this.lowerCaseFlag = config.lowerCaseFlag;
    this.maximumNumberOfWords = config.maximumNumberOfWords;
    this.numberOfTexts = config.numberOfTexts;
    this.characterLevelFlag = config.characterLevelFlag;
    this.defaultToken = config.defaultToken;
  }

  /** 
   * @param corpus Array of texts as strings.
   * @param hashingFlag Boolean which indicates whether hashing should be used
   * @param mode Statistic to be inserted in matrix cells.
   * @returns Array of arrays of integers interpretable as an occurence matrix,
   * with rows corresponding to texts and columns corresponding to features.
   */

  createOccurrenceMatrixFromStrings(corpus: string[], hashingFlag = false, 
    spaceFactor: number,
    mode: string, outputArrays: boolean): number[][] | tfc.Tensor<tfc.Rank.R2> {

    if (!hashingFlag) {

      let tokenizer: TextTokenization;
      let enumerator: TypeEnumeration;
      let digitalizer: TokenDigitalization;
      let transformer: SequenceTransformation;
      let tokensByText: string[][];
      let corpusTokens: string[];
      let corpusTypes: string[];
      let typeCountsInCorpus: Map<string, number>;
      let typesSortedByCount: string[];
      let typeIndex: Map<string, number>;
      let corpusTokenSequences: string[][];
      const integerSequences: number[][] = [];
      let generator: IterableIterator<number[]>;

      let dataTensor: tfc.Tensor<tfc.Rank.R2>;
      let dataMatrix: number[][];

      tokenizer = new TextTokenization(corpus, this.config);
      enumerator = new TypeEnumeration(corpus, this.config);
      digitalizer = new TokenDigitalization(corpus, this.config);
      transformer = new SequenceTransformation(this.config);
      const result0 = tokenizer.collectTokens(corpus);
      tokensByText = result0[0];
      corpusTokens = result0[1];
      const result1 = tokenizer.collectTypes(tokensByText, corpusTokens);
      corpusTypes = result1[1];
      typeCountsInCorpus =
        enumerator.countTypesInCorpus(corpusTokens, corpusTypes);
      typesSortedByCount = enumerator.sortTypesByCount(typeCountsInCorpus);
      typeIndex = enumerator.createTypeIndex(
        typeCountsInCorpus, typesSortedByCount, this.config.defaultToken);
      corpusTokenSequences =
        tokenizer.createCorpusTokenSequences(corpus, this.config);
      generator =
        digitalizer.generateIntegerSequencesFromCounts(typeIndex, this.config,
          corpusTokenSequences);

      /*for (const s of generator) {
        integerSequences.push(s);
      }*/
      let quit = false;
      let retrieve = generator.next();
      do {
        integerSequences.push(retrieve.value);
        retrieve = generator.next();
        quit = retrieve.done;
      } while (quit === false);

      const returnValue = transformer.createOccurrenceMatrix(this.config,
        integerSequences, mode, outputArrays);
      if (returnValue instanceof Array) {
        dataMatrix = returnValue as number[][];
      }
      else {
        dataTensor = returnValue;
      }
      if (outputArrays) {
        return dataMatrix;
      }
      else {
        return dataTensor;
      }
    }
    else {

      let integerSequences: number[][] = [];

      let dataTensor: tfc.Tensor<tfc.Rank.R2>;
      let dataMatrix: number[][];

      const digitalizer = new TokenDigitalization(corpus, this.config);
      const transformer = new SequenceTransformation(this.config);

      const hashAwaiter = async () => {
        integerSequences = await digitalizer.hashWordsToIntegers(
          corpus, spaceFactor, this.config);

        const returnValue = transformer.createOccurrenceMatrix(this.config,
          integerSequences, mode, outputArrays);
        if (returnValue instanceof Array) {
          dataMatrix = returnValue as number[][];
        }
        else {
          dataTensor = returnValue;
        }

      };
      hashAwaiter();

      if (outputArrays) {
        return dataMatrix;
      }
      else {
        return dataTensor;
      }
    }
  }

  /**
   * @param corpus Arrays of strings representing texts.
   * @param windowSize Length of the subsequences to be processed.
   * @param panePosition Position in the window of the token the context of
   * which is to be extracted.
   * @param hashingFlag Boolean which indicates whether hashing should be used.
   * @returns Array of arrays of integers which are the contexts of integers in
   * sequences and an array of those corresponding integers
   */
  createSubsequencesFromStrings(corpus: string[],
    windowSize: number,
    panePosition: number,
    hashingFlag = false,
    outputArrays = false
  ): [number[][], number[]] | [tfc.Tensor<tfc.Rank.R2>,
    tfc.Tensor<tfc.Rank.R1>] {
    let contextCollector: number[][];
    let objectInContextCollector: number[];
    let contextTensor: tfc.Tensor<tfc.Rank.R2>;
    let objectInContextTensor: tfc.Tensor<tfc.Rank.R1>;

    if (!hashingFlag) {

      let tokensByText: string[][];
      let corpusTokens: string[];
      let corpusTypes: string[];
      let typeCountsInCorpus: Map<string, number>;
      let typesSortedByCount: string[];
      let typeIndex: Map<string, number>;
      let corpusTokenSequences: string[][];
      const integerSequences: number[][] = [];
      let generator: IterableIterator<number[]>;
      const tokenizer = new TextTokenization(this.corpus, this.config);
      const enumerator = new TypeEnumeration(this.corpus, this.config);
      const digitalizer = new TokenDigitalization(this.corpus, this.config);
      const transformer = new SequenceTransformation(this.config);
      const result0 = tokenizer.collectTokens(this.corpus);
      tokensByText = result0[0];
      corpusTokens = result0[1];
      const result1 = tokenizer.collectTypes(tokensByText, corpusTokens);
      corpusTypes = result1[1];
      typeCountsInCorpus =
        enumerator.countTypesInCorpus(corpusTokens, corpusTypes);
      typesSortedByCount = enumerator.sortTypesByCount(typeCountsInCorpus);
      typeIndex = enumerator.createTypeIndex(
        typeCountsInCorpus, typesSortedByCount, this.config.defaultToken);
      corpusTokenSequences =
        tokenizer.createCorpusTokenSequences(this.corpus, this.config);
      generator =
        digitalizer.generateIntegerSequencesFromCounts(typeIndex, this.config,
          corpusTokenSequences);

      /*for (const s of generator) {
        integerSequences.push(s);
      }*/
      let quit = false;
      let retrieve = generator.next();
      do {
        integerSequences.push(retrieve.value);
        retrieve = generator.next();
        quit = retrieve.done;
      } while (quit === false);

      const result = transformer.createSubsequences(integerSequences,
        windowSize,
        panePosition,
        outputArrays
      );
      if (outputArrays) {
        contextCollector = result[0] as number[][];
        objectInContextCollector = result[1] as number[];
      }
      else {
        contextTensor = result[0] as tfc.Tensor<tfc.Rank.R2>;
        objectInContextTensor = result[1] as tfc.Tensor<tfc.Rank.R1>;
      }

    }
    else {

      let integerSequences: number[][] = [];

      const digitalizer = new TokenDigitalization(this.corpus, this.config);
      const transformer = new SequenceTransformation(this.config);

      const hashAwaiter = async () => {
        integerSequences = await digitalizer.hashWordsToIntegers(
          this.corpus, spaceFactor, this.config);

        const result1 = transformer.createSubsequences(integerSequences,
          windowSize,
          panePosition,
          outputArrays
        );

        if (outputArrays) {
          contextCollector = result1[0] as number[][];
          objectInContextCollector = result1[1] as number[];
        }
        else {
          contextTensor = result1[0] as tfc.Tensor<tfc.Rank.R2>;
          objectInContextTensor = result1[1] as tfc.Tensor<tfc.Rank.R1>;
        }

        if (outputArrays) {
          return [contextCollector, objectInContextCollector];
        }
        else {

          return [contextTensor, objectInContextTensor];
        }
      };
      hashAwaiter();
    }
    if (outputArrays) {
      return [contextCollector, objectInContextCollector];
    }
    else {
      return [contextTensor, objectInContextTensor];
    }
  }
}
