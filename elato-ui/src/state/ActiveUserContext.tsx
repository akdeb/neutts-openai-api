import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api } from "../api";

type User = {
  id: string;
  name: string;
  age?: number | null;
  dob?: string | null;
  hobbies: string[];
  personality_type?: string | null;
  likes: string[];
  current_personality_id?: string | null;
  user_type?: string | null;
  device_volume?: number | null;
};

type ActiveUserContextValue = {
  users: User[];
  activeUserId: string | null;
  activeUser: User | null;
  setActiveUserId: (id: string | null) => Promise<void>;
  refreshUsers: () => Promise<void>;
};

const ActiveUserContext = createContext<ActiveUserContextValue | null>(null);

const STORAGE_KEY = "elato.activeUserId";

export function ActiveUserProvider({ children }: { children: React.ReactNode }) {
  const [users, setUsers] = useState<User[]>([]);
  const [activeUserId, setActiveUserIdState] = useState<string | null>(null);

  const refreshUsers = useCallback(async () => {
    const data = await api.getUsers();
    setUsers(data);
  }, []);

  const setActiveUserId = useCallback(async (id: string | null) => {
    setActiveUserIdState(id);

    if (id) {
      localStorage.setItem(STORAGE_KEY, id);
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }

    try {
      await api.setActiveUser(id);
    } catch {
      // ignore: UI will still function, but ESP32 side will keep old value
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    const init = async () => {
      try {
        await refreshUsers();

        const stored = localStorage.getItem(STORAGE_KEY);
        const server = await api.getActiveUser().catch(() => null);

        const candidate = stored || server?.user_id || null;

        if (cancelled) return;

        if (candidate) {
          setActiveUserIdState(candidate);
          await api.setActiveUser(candidate).catch(() => null);
        } else {
          // If no active user yet, default to the first existing user
          const currentUsers = await api.getUsers().catch(() => []);
          if (currentUsers.length > 0) {
            const first = currentUsers[0]?.id;
            if (first) {
              setActiveUserIdState(first);
              localStorage.setItem(STORAGE_KEY, first);
              await api.setActiveUser(first).catch(() => null);
            }
          }
        }
      } catch {
        // ignore
      }
    };

    init();
    return () => {
      cancelled = true;
    };
  }, [refreshUsers]);

  const activeUser = useMemo(() => {
    if (!activeUserId) return null;
    return users.find((u) => u.id === activeUserId) || null;
  }, [users, activeUserId]);

  const value = useMemo(
    () => ({ users, activeUserId, activeUser, setActiveUserId, refreshUsers }),
    [users, activeUserId, activeUser, setActiveUserId, refreshUsers]
  );

  return <ActiveUserContext.Provider value={value}>{children}</ActiveUserContext.Provider>;
}

export function useActiveUser() {
  const ctx = useContext(ActiveUserContext);
  if (!ctx) throw new Error("useActiveUser must be used within ActiveUserProvider");
  return ctx;
}
