// API configuration
// Uses environment variable or falls back to localhost for development

export const API_URL = process.env.EXPO_PUBLIC_API_URL || 'http://192.168.0.159:8080';

export const apiEndpoints = {
  analyze: `${API_URL}/analyze`,
  // Add other endpoints here as needed
};
