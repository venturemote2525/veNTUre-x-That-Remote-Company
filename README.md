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

### 3. Create .env.local file
Add environment variables to `.env.local` file.
```env
EXPO_PUBLIC_SUPABASE_URL=
EXPO_PUBLIC_SUPABASE_ANON_KEY=
```

### 4. Add `.aar` file
Place the `.aar` file in `android/app/libs` folder.`

### 5. Ensure Java JDK 17 installed
Add the JDK path in `android/gradle.properties`
```properties
org.gradle.java.home=/path/to/your/jdk-17
```

### 6. Run the project
- Run `npm install` to install dependencies
- Run `npx expo start` to start the project with expo go app
- Run `npx expo run:android` to run the project on android device