import socket

def start_stage_1_server(host='127.0.0.1', port=9999):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Allows rapid reuse of the socket port if you restart the script quickly
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(self_reply := f"Listening on {host}:{port}... Launch Minecraft and enter a world.")

    conn, addr = server_socket.accept()
    print(f"Connected successfully by client at: {addr}")

    buffer = ""
    tick_count = 0

    try:
        while True:
            data = conn.recv(1024).decode('utf-8')
            if not data:
                print("Client disconnected.")
                break
            
            buffer += data
            # Handle multiple streams or partial packets split by a newline
            while "\n" in buffer:
                packet, buffer = buffer.split("\n", 1)
                if packet.strip():
                    tick_count += 1
                    print(f"[Tick {tick_count:05d}] Recv: {packet}")
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        conn.close()
        server_socket.close()

if __name__ == "__main__":
    start_stage_1_server()