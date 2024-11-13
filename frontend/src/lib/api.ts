const apiUrl = import.meta.env.VITE_API_URL;

export const api = {
  parseTrades: async (file: File) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${apiUrl}/parse-trades/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to parse trades');
      }

      return await response.json();
    } catch (error) {
      console.error('Error parsing trades:', error);
      throw error;
    }
  },
};
