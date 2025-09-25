import { createContext, useContext, useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import { Session, User } from '@supabase/supabase-js';
import { UserProfile } from '@/types/database-types';

type AuthContextType = {
  session: Session | null;
  user: User | null;
  profile: UserProfile | null;
  loading: boolean;
  profileLoading: boolean;
  authenticated: boolean;
  refreshProfile: () => void;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const [session, setSession] = useState<Session | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [authenticated, setAuthenticated] = useState(false);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [profileLoading, setProfileLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [initialised, setInitialised] = useState(false);

  useEffect(() => {
    if (initialised) return;
    const checkAuthStatus = async () => {
      try {
        // Check for existing session
        const {
          data: { session },
          error: sessionError,
        } = await supabase.auth.getSession();
        if (sessionError || !session?.user) {
          console.log('No active session');
          setUser(null);
          setSession(null);
          setAuthenticated(false);
          setLoading(false);
          setInitialised(true);
          return;
        }
        // Validate user token
        const {
          data: { user },
          error: userError,
        } = await supabase.auth.getUser();
        if (userError || !user) {
          console.log('User validation failed: ', userError);
          setUser(null);
          setSession(null);
          setAuthenticated(false);
          setLoading(false);
          setInitialised(true);
          return;
        }
        console.log('Valid session');
        setUser(user);
        setSession(session);
        setAuthenticated(true);
        setLoading(false);
        setInitialised(true);
        return;
      } catch (error) {
        console.log('Failed to initialise auth: ', error);
        setUser(null);
        setSession(null);
        setAuthenticated(false);
        setLoading(false);
        setInitialised(true);
      }
    };
    if (!initialised) {
      checkAuthStatus();
    }

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session ?? null);
      setUser(session?.user ?? null);
      setAuthenticated(!!session?.user);
      console.log('Auth event: ', _event);
    });
    return () => subscription.unsubscribe();
  }, []);

  const fetchProfile = async () => {
    if (!user) {
      setProfile(null);
      setProfileLoading(false);
      return;
    }
    setProfileLoading(true);
    const { data, error } = await supabase
      .from('user_profiles')
      .select('*')
      .eq('user_id', user.id)
      .order('created_at',{ascending: false})
      .limit(1);
    if (error) {
      console.log('Error fetching profile: ', error);
      setProfile(null);
    } else {
      console.log('fetched: ', data);
      setProfile(data);
    }
    setProfileLoading(false);
  };

  useEffect(() => {
    if (user) fetchProfile();
  }, [user]);

  const refreshProfile = async () => {
    if (!user) return;
    setProfileLoading(true);
    const { data, error } = await supabase
      .from('user_profiles')
      .select('*')
      .eq('user_id', user.id)
      .order('created_at', {ascending: false})
      .limit(1);

    if (error) {
      console.log('Error refreshing profile: ', error);
      setProfile(null);
    } else {
      setProfile(data);
    }
    setProfileLoading(false);
  };

  return (
    <AuthContext.Provider
      value={{
        session,
        user,
        profile,
        loading,
        profileLoading,
        authenticated: !!user && !!authenticated,
        refreshProfile,
      }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within AuthProvider');
  return context;
};
