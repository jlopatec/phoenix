import {
  formatJSONString,
  isJSONObjectString,
  safelyParseJSONObject,
} from "../jsonUtils";

describe("isJSONObjectString", () => {
  it("detects invalid JSON and JSON that's not objects", () => {
    expect(isJSONObjectString("")).toEqual(false);
    expect(isJSONObjectString("123")).toEqual(false);
    expect(isJSONObjectString("[]")).toEqual(false);
    expect(isJSONObjectString("[1, 2, 3]")).toEqual(false);
    expect(isJSONObjectString("{")).toEqual(false);
    expect(isJSONObjectString("{]")).toEqual(false);
    expect(isJSONObjectString("{1: 2}")).toEqual(false);
    expect(isJSONObjectString("{1, 2}")).toEqual(false);
    expect(isJSONObjectString("{1: 2, 3}")).toEqual(false);
  });
  it("detects valid JSON objects", () => {
    expect(isJSONObjectString("{}")).toEqual(true);
    expect(isJSONObjectString('{"a": 1}')).toEqual(true);
    expect(isJSONObjectString('{"a": "b"}')).toEqual(true);
    expect(isJSONObjectString('{"a": {"b": 1}}')).toEqual(true);
    expect(isJSONObjectString('{"a": [1, 2, 3]}')).toEqual(true);
  });
});

describe("formatJSONString", () => {
  it("pretty prints valid JSON", () => {
    expect(formatJSONString('{"a":1}')).toEqual('{\n  "a": 1\n}');
  });
  it("falls back to the raw value for invalid JSON", () => {
    expect(formatJSONString('{"a":')).toEqual('{"a":');
    expect(formatJSONString(undefined)).toEqual("");
  });
});

describe("safelyParseJSONObject", () => {
  it("parses JSON objects", () => {
    expect(safelyParseJSONObject('{"a":1}')).toEqual({ a: 1 });
  });
  it("returns null for anything else", () => {
    expect(safelyParseJSONObject("not json")).toBeNull();
    expect(safelyParseJSONObject("[1]")).toBeNull();
    expect(safelyParseJSONObject("1")).toBeNull();
    expect(safelyParseJSONObject({ a: 1 })).toBeNull();
  });
});
