/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./app/**/*.{js,jsx,ts,tsx}'],
  presets: [require('nativewind/preset')],
  theme: {
    extend: {
      // Define colours here
      colors: {
        background: '#f2f2f2',
      },
      // Define fonts here
      fontFamily: {
        heading: ['Poppins-Bold', 'sans-serif'],
        body: ['Poppins-Regular', 'sans-serif'],
        bodyBold: ['Poppins-Medium', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
