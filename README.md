# veNTUre-x-That-Remote-Company

## Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/venturemote2525/veNTUre-x-That-Remote-Company.git
```

### 2. Switch to `dev` branch
```bash
git checkout dev
```

### 3. Create environment files
**`.env.local`**
- Add environment variables to `.env.local` at the project root
```env
EXPO_PUBLIC_SUPABASE_URL=
EXPO_PUBLIC_SUPABASE_ANON_KEY=
```
**`local.properties`**
- Add your Android SDK location inside `android/local.properties`

macOS / Linux:
```properties
sdk.dir=/Users/your-username/Library/Android/sdk
```
Windows:
```properties
sdk.dir=C:\\Users\\YourUsername\\AppData\\Local\\Android\\Sdk
```

### 4. Add `.aar` file
Place the `.aar` file (from the weighing scale SDK) in `android/app/libs` (Create the `libs` folder if it doesn't exist).

### 5. Ensure Java JDK 17 installed
Add the JDK path in `android/gradle.properties`
```properties
org.gradle.java.home=/path/to/your/jdk-17
```

### 6. Install dependencies
```bash
cd frontend
npm install
```

### 7. Run the project
- Run `npx expo start` to start the project with Expo Go app
- Run `npx expo run:android` to run the project on android device

## Project Overview

### Folder Structure
```
project/
│
├─ backend/
│
└─ frontend/
    ├─ android/     # Android files for SDK
    ├─ app/         # App screens
    ├─ assets/      # Images, fonts, etc.
    ├─ components/  # Reusable UI components
    ├─ constants/   # Static configuration values
    ├─ context/     # Global state providers
    ├─ hooks/       # Custom React hooks
    ├─ ios/         # IOS files for SDK
    ├─ lib/         # Supabase configuration
    ├─ types/       # Database and SDK types
    └─ utils/       # Helper and API utility functions

```

### Key Components
| Component                                 | Description                                   |
|-------------------------------------------|-----------------------------------------------|
| CustomAlert (`ui/CustomAlert.tsx`)        | Custom modal for confirmation and alerts      |
| Gluestack UI (`ui/gluestack-ui-provider`) | UI library for theming and consistent design  |
| LoadingScreen (`ui/LoadingScreen.tsx`)    | Custom loading screen shown during data fetch |
| Themed (`ui/Themed.tsx`)                  | Custom themed Text/View/SafeAreaView wrappers |

### Contexts
| Context                                         | Description                                |
|-------------------------------------------------|--------------------------------------------|
| AuthContext (`context/AuthContext.tsx`)         | Handle login/logout, authentication status |
| UserContext (`context/UserContext.tsx`)         | Context provider for user data             |
| ThemeContext (`context/ThemeContext.tsx`)       | Manages app theme                          |
| ICDeviceContext (`context/ICDeviceContext.tsx`) | Handle connection to SDK                   |