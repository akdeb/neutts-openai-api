import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { useActiveUser } from '../state/ActiveUserContext';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import { useEffect, useState } from 'react';

export const Layout = () => {
  const { activeUser } = useActiveUser();
  const navigate = useNavigate();
  const [activePersonalityName, setActivePersonalityName] = useState<string | null>(null);
  const [deviceConnected, setDeviceConnected] = useState<boolean>(false);
  const [deviceSessionId, setDeviceSessionId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const ds = await api.getDeviceStatus().catch(() => ({ connected: false, session_id: null }));
        if (!cancelled) {
          setDeviceConnected(!!ds?.connected);
          setDeviceSessionId(ds?.session_id || null);
        }

        const selectedId = activeUser?.current_personality_id;
        if (!selectedId) {
          if (!cancelled) setActivePersonalityName(null);
          return;
        }

        const ps = await api.getPersonalities(true).catch(() => []);
        const selected = ps.find((p: any) => p.id === selectedId);
        if (!cancelled) setActivePersonalityName(selected?.name || null);
      } catch {
        // ignore
      }
    };

    load();
    const id = window.setInterval(load, 1500);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [activeUser?.current_personality_id]);

  return (
    <div className="flex h-screen overflow-hidden bg-[#f6f0e6]">
      <Sidebar />
      <main className="flex-1 min-h-0 p-8 pb-36 overflow-y-auto retro-dots">
        <div className="max-w-4xl mx-auto">
          <Outlet />
        </div>

        {activeUser?.current_personality_id && (
          <div className="fixed bottom-0 left-64 right-0 pointer-events-none">
            <div className="max-w-4xl mx-auto px-8 pb-6 pointer-events-auto">
              <div className="bg-gray-50 border-2 rounded-2xl px-4 py-3 flex items-center justify-between shadow-2xl">
                <div>
                  <div className="font-mono text-xs text-gray-600">
                    Active: <span className="font-bold text-black">{activePersonalityName || '—'}</span>
                  </div>
                  <div className="font-mono text-[11px] text-gray-500 mt-1">
                    {deviceConnected ? 'Chat in progress' : 'Ready to connect'}
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    className="retro-btn px-4 py-2 text-sm"
                    onClick={() => navigate('/test')}
                  >
                    Test
                  </button>
                  {deviceConnected && deviceSessionId && (
                    <button
                      type="button"
                      className="retro-btn bg-white px-4 py-2 text-sm"
                      onClick={() => navigate(`/conversations?session=${encodeURIComponent(deviceSessionId)}`)}
                    >
                      View
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};
