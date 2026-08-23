import { describe, expect, it } from "vitest";
import { compactNumber, dollars, percent, seconds } from "./format";

describe("measurement formatting", () => {
  it("formats evaluation values for the reviewer", () => {
    expect(percent(0.5194)).toBe("51.9%");
    expect(seconds(1512.3)).toBe("1.51s");
    expect(dollars(0.002632)).toBe("$0.0026");
    expect(compactNumber(2711)).toBe("2.7K");
  });

  it("does not invent a cost when the provider did not report one", () => {
    expect(dollars(null)).toBe("N/A");
  });
});
