/**
 * Checks if a string is a valid JSON string.
 * @param {Object} options - The options object.
 * @param {string} options.str - The string to check.
 * @param {boolean} [options.excludePrimitives=false] - Whether to exclude primitive types from the check.
 * @param {boolean} [options.excludeArray=false] - Whether to exclude arrays from the check.
 * @returns {boolean} - Returns true if the string is a valid JSON string, false otherwise.
 */
export function isJSONString({
  str,
  excludePrimitives = false,
  excludeArray = false,
}: {
  str: string;
  excludePrimitives?: boolean;
  excludeArray?: boolean;
}) {
  try {
    const parsed = JSON.parse(str);
    if (excludePrimitives && typeof parsed !== "object") {
      return false;
    }
    if (excludeArray && Array.isArray(parsed)) {
      return false;
    }
  } catch (e) {
    return false;
  }
  return true;
}

export function isJSONObjectString(str: string) {
  return isJSONString({ str, excludeArray: true, excludePrimitives: true });
}

/**
 * Pretty prints a JSON string, falling back to the raw value if it is not valid JSON.
 */
export function formatJSONString(str: unknown): string {
  if (typeof str !== "string") {
    return String(str ?? "");
  }
  try {
    return JSON.stringify(JSON.parse(str), null, 2);
  } catch (e) {
    return str;
  }
}

/**
 * Parses a JSON object string, returning null if it is not a valid JSON object.
 */
export function safelyParseJSONObject(
  str: unknown
): Record<string, unknown> | null {
  if (typeof str !== "string" || !isJSONObjectString(str)) {
    return null;
  }
  return JSON.parse(str);
}
