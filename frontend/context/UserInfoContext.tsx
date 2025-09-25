import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useAuth } from '@/context/AuthContext';

export interface UserInfo {
  age: number;
  height: number;
  sex: number; // 1 = Male, 2 = Female
  peopleType: number; // 0 = Normal, 1 = Sportsman
  status: string;
  persistent: boolean;
}

interface UserInfoContextType {
  userInfo: UserInfo | null;
  loading: boolean;
  refreshUserInfo: () => void;
}

const UserInfoContext = createContext<UserInfoContextType | undefined>(undefined);

interface UserInfoProviderProps {
  children: ReactNode;
}

export const UserInfoProvider: React.FC<UserInfoProviderProps> = ({ children }) => {
  const { profile, profileLoading } = useAuth();
  const [userInfo, setUserInfo] = useState<UserInfo | null>(null);
  const [loading, setLoading] = useState(true);

  const calculateAge = (dobString: string | null | undefined): number => {
    if (!dobString) return 25; // default age
    const dob = new Date(dobString);
    const today = new Date();
    let age = today.getFullYear() - dob.getFullYear();
    const monthDiff = today.getMonth() - dob.getMonth();
    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < dob.getDate())) {
      age--;
    }
    return age > 0 ? age : 25;
  };

  const getSexCode = (gender: string | null | undefined): number => {
    if (!gender) return 1; // default to male
    return gender.toLowerCase() === 'female' ? 2 : 1;
  };

  const processUserInfo = () => {
    setLoading(true);

    if (profile && !profileLoading) {
      const processedInfo: UserInfo = {
        age: calculateAge(profile.dob),
        height: profile.height || 170,
        sex: getSexCode(profile.gender),
        peopleType: 0, // default to normal person type
        status: 'success',
        persistent: true
      };

      console.log('Processed user info from profile:', processedInfo);
      setUserInfo(processedInfo);
    } else {
      // Set defaults if no profile data
      const defaultInfo: UserInfo = {
        name: "User",
        age: 25,
        height: 170,
        sex: 1,
        peopleType: 0,
        status: 'default',
        persistent: false
      };

      console.log('Using default user info');
      setUserInfo(defaultInfo);
    }

    setLoading(false);
  };

  // Process user info when profile changes
  useEffect(() => {
    processUserInfo();
  }, [profile, profileLoading]);

  const refreshUserInfo = () => {
    processUserInfo();
  };

  return (
    <UserInfoContext.Provider value={{ userInfo, loading, refreshUserInfo }}>
      {children}
    </UserInfoContext.Provider>
  );
};

export const useUserInfo = (): UserInfoContextType => {
  const context = useContext(UserInfoContext);
  if (!context) {
    throw new Error('useUserInfo must be used within a UserInfoProvider');
  }
  return context;
};