import { getAttributeMessages } from "../types";

describe("getAttributeMessages", () => {
  it("extracts messages from well-formed attributes", () => {
    expect(
      getAttributeMessages([
        { message: { role: "user", content: "hi" } },
        { message: { role: "assistant", content: "hello" } },
      ])
    ).toEqual([
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ]);
  });

  it("returns an empty list for non-array values", () => {
    expect(getAttributeMessages("not messages")).toEqual([]);
    expect(getAttributeMessages({ message: { role: "user" } })).toEqual([]);
    expect(getAttributeMessages(null)).toEqual([]);
    expect(getAttributeMessages(undefined)).toEqual([]);
  });

  it("drops malformed entries", () => {
    expect(
      getAttributeMessages([
        null,
        "oops",
        { message: "oops" },
        { message: { role: "user", content: "hi" } },
      ])
    ).toEqual([{ role: "user", content: "hi" }]);
  });
});
