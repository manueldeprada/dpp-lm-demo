import test from 'node:test';
import assert from 'node:assert/strict';
import { splitTwoLines } from '../js/labels-lib.js';

test('empty string gives one empty line', () => assert.deepEqual(splitTwoLines(''), ['']));
test('whitespace only gives one empty line', () => assert.deepEqual(splitTwoLines('  \n '), ['']));
test('one word stays on one line', () => assert.deepEqual(splitTwoLines('hello'), ['hello']));
test('two words split between them', () => assert.deepEqual(splitTwoLines('hello world'), ['hello', 'world']));
test('odd word count', () => assert.deepEqual(splitTwoLines('a bb ccc dddd eeeee'), ['a bb ccc', 'dddd eeeee']));
test('even word count', () => assert.deepEqual(splitTwoLines('aaa bbb ccc ddd'), ['aaa bbb', 'ccc ddd']));
test('a tie takes the earlier boundary', () => {
  assert.deepEqual(splitTwoLines('a b c d'), ['a b', 'c d']);
  assert.deepEqual(splitTwoLines('aaaa b c'), ['aaaa', 'b c']);   // 4 vs 3 beats 6 vs 1
  assert.deepEqual(splitTwoLines('aa b bb'), ['aa', 'b bb']);     // 2 vs 4 and 4 vs 2 tie: earlier
});
test('a long first word stays alone on line one', () => assert.deepEqual(splitTwoLines('supercalifragilistic is fun'), ['supercalifragilistic', 'is fun']));
test('whitespace is collapsed and trimmed', () => assert.deepEqual(splitTwoLines('  a \n b\t'), ['a', 'b']));
test('joining with a space gives back the normalized input', () => {
  for (const t of ['one', 'one two', 'The quick brown fox jumps over the lazy dog', 'x yy zzz wwww', 'a  b   c']) {
    const norm = t.replace(/\s+/g, ' ').trim();
    const r = splitTwoLines(t);
    assert.ok(r.length <= 2);
    assert.equal(r.join(' '), norm);
  }
});
