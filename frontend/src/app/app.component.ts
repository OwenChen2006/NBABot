import { Component } from '@angular/core';
import { ChatService } from './services/chat.service';

export interface Message {
  sender: 'user' | 'bot';
  text: string;
  evidence?: any[];
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'AI Engineering Sandbox';
  messages: Message[] = [];
  userInput = '';
  isLoading = false;


  constructor(private chatService: ChatService) {}

  sendMessage(): void {
    const input = this.userInput.trim();
    if (!input) {
      return;
    }
    this.messages.push({ sender: 'user', text: input });
    this.userInput = '';
    this.isLoading = true;
    this.chatService.sendMessage(input).subscribe({
      next: (res: any) => {
        const reply = res?.reply ?? res?.answer ?? 'No Answer.';
        this.messages.push({ sender: 'bot', text: reply, evidence: res?.evidence });
        this.isLoading = false;
      },
      error: () => {
        this.messages.push({ sender: 'bot', text: 'Error contacting server.' });
        this.isLoading = false;
      }
    });
  }
}